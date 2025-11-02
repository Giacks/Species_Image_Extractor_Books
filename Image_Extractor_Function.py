# FUNCTION: extract_fish_from_pdf

def extract_fish_from_pdf(PDF_PATH: str, OUT_DIR: str, ZIP_PATH: str, DPI: int = 400):
    
    import os, re, zipfile, cv2, numpy as np, fitz
    from pdf2image import convert_from_path
    from PIL import Image


    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n📘 Processing {PDF_PATH}")
    print(f"→ Output directory: {OUT_DIR}")
    print(f"→ ZIP archive: {ZIP_PATH}")

    # ============================================================
    # === Italic detection helpers ===============================
    # ============================================================

    def fontname_looks_italic(fontname: str) -> bool:
        if not fontname:
            return False
        fn = fontname.lower()
        fn = re.sub(r"^[A-Z]{6}\+", "", fn)
        return any(k in fn for k in ["italic", "oblique", "slanted", "it", "ita"])

    def detect_text_slant_angle(img_gray, bbox, expand=2):
        x0, y0, w, h = bbox
        ex = int(min(w, h) * 0.2) * expand
        x0e = max(0, int(x0 - ex))
        y0e = max(0, int(y0 - ex))
        x1e = min(img_gray.shape[1], int(x0 + w + ex))
        y1e = min(img_gray.shape[0], int(y0 + h + ex))
        crop = img_gray[y0e:y1e, x0e:x1e]
        if crop.size == 0:
            return None
        blur = cv2.GaussianBlur(crop, (3, 3), 0)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(bw) > 127:
            bw = 255 - bw
        edges = cv2.Canny(bw, 50, 150)
        ys, xs = np.nonzero(edges)
        if len(xs) < 30:
            return None
        coords = np.vstack([xs, ys]).astype(np.float64)
        coords -= coords.mean(axis=1, keepdims=True)
        U, S, Vt = np.linalg.svd(coords @ coords.T)
        principal = U[:, 0]
        dx, dy = principal[0], principal[1]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        if angle_deg > 90:
            angle_deg -= 180
        if angle_deg <= -90:
            angle_deg += 180
        return angle_deg

    def is_italic_span(span: dict, page_img_gray=None, scale=1.0, angle_thresh=3.0):
        fontname = span.get("font", "")
        if fontname_looks_italic(fontname):
            return True
        if page_img_gray is not None and "bbox" in span:
            x0, y0, x1, y1 = span["bbox"]
            x0 *= scale; y0 *= scale; x1 *= scale; y1 *= scale
            bbox_px = (x0, y0, x1 - x0, y1 - y0)
            angle = detect_text_slant_angle(page_img_gray, bbox_px)
            if angle is not None and abs(angle) >= angle_thresh:
                return True
        return False

    # ============================================================
    # === Process each page ======================================
    # ============================================================

    doc = fitz.open(PDF_PATH)
    page_count = len(doc)
    scale = DPI / 72.0
    rendered_pages = convert_from_path(PDF_PATH, dpi=DPI)

    all_matches = []

    for page_index, (page_image, page_fitz) in enumerate(zip(rendered_pages, doc), start=1):
        print(f"\n=== Page {page_index}/{page_count} ===")
        page_dir = os.path.join(OUT_DIR, f"page_{page_index:03d}")
        os.makedirs(page_dir, exist_ok=True)

        page_path = os.path.join(page_dir, f"page_{page_index:03d}.jpg")
        page_image.convert("RGB").save(page_path, "JPEG", quality=95)

        img_cv = cv2.imread(page_path)
        page_img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # --- Extract italic spans ---
        blocks = page_fitz.get_text("dict")["blocks"]
        italic_boxes, species_names = [], []

        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    if re.match(r"^[A-Z][a-z]+ [a-z]+$", text):
                        if is_italic_span(span, page_img_gray=page_img_gray, scale=scale):
                            x0, y0, x1, y1 = span["bbox"]
                            x0 *= scale; y0 *= scale; x1 *= scale; y1 *= scale
                            italic_boxes.append((x0, y0, x1 - x0, y1 - y0))
                            species_names.append(text.replace(" ", "_"))

        print(f" → Detected {len(species_names)} italic species names.")

        # --- Detect fish contours ---
        blur = cv2.medianBlur(page_img_gray, 5)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = page_img_gray.shape
        page_area = w * h
        fish_boxes = []
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            if ww * hh > page_area * 0.002:
                fish_boxes.append((x, y, ww, hh))
        print(f" → Found {len(fish_boxes)} fish-like regions.")

        # --- Match captions to fish boxes ---
        print("Matching fish boxes to nearest captions below them...")
        matches = []
        used_species, used_boxes = set(), set()
        for fi, (fx, fy, fw, fh) in enumerate(fish_boxes):
            fish_bottom = fy + fh
            fish_center_x = fx + fw / 2
            below_candidates = []
            for si, (tx, ty, tw, tht) in enumerate(italic_boxes):
                if si in used_species:
                    continue
                if ty > fish_bottom and (ty - fish_bottom) < 500:
                    horizontal_distance = abs(tx - fx)
                    vertical_distance = abs(ty - fish_bottom)
                    distance_score = vertical_distance + 0.5 * horizontal_distance
                    below_candidates.append((distance_score, si, tx, ty, tw, tht))
            if below_candidates:
                below_candidates.sort(key=lambda x: x[0])
                _, si, tx, ty, tw, tht = below_candidates[0]
                species = species_names[si]
                matches.append((species, (fx, fy, fw, fh)))
                used_species.add(si)
                used_boxes.add(fi)

        remaining_species = [s for i, s in enumerate(species_names) if i not in used_species]
        remaining_fish = [b for i, b in enumerate(fish_boxes) if i not in used_boxes]
        for s, b in zip(remaining_species, remaining_fish):
            matches.append((s, b))

        print(f" → Matched {len(matches)} species to fish images.")

        # --- Crop and save ---
        pad = 25
        for species, (x, y, ww, hh) in matches:
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + ww + pad)
            y1 = min(h, y + hh + pad)
            crop = img_cv[int(y0):int(y1), int(x0):int(x1)]
            species_dir = os.path.join(page_dir, species)
            os.makedirs(species_dir, exist_ok=True)
            out_path = os.path.join(species_dir, f"{species}_p{page_index:03d}.jpg")
            cv2.imwrite(out_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            all_matches.append(out_path)

    # ============================================================
    # === Zip everything =========================================
    # ============================================================

    print("\nZipping all results...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(OUT_DIR):
            for f in files:
                if f.lower().endswith(".jpg"):
                    absf = os.path.join(root, f)
                    relf = os.path.relpath(absf, OUT_DIR)
                    zipf.write(absf, relf)

    print(f"\n✅ Done. ZIP created: {ZIP_PATH}")
    return all_matches


# Example direct call (you can comment this out if using as a module)
#if __name__ == "__main__":
#    extract_fish_from_pdf(
#        PDF_PATH="2._wiof_volume_2_colour_plates.pdf",
#        OUT_DIR="wiof_output_allpages",
#        ZIP_PATH="wiof_output_allpages.zip",
#    )
