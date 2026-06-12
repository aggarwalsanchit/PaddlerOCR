"""
preprocessor.py
Exact port of your preprocess_image() and detect_regions() functions.
Produces _gray.png and _otsu.png variants for every image.
"""
import os
import cv2
import numpy as np
from config import (
    UPSCALE_FACTOR, BILATERAL_D, BILATERAL_SC, BILATERAL_SS,
    CLAHE_CLIP, CLAHE_TILE, MIN_AREA, MIN_W, MIN_H, PAD,
)


def preprocess_image(image_path: str):
    """
    Exact port of your preprocess_image() function.
    Returns (original_bgr, [gray_enhanced, otsu_binary])
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    original = image.copy()

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4x upscale
    gray = cv2.resize(gray, None,
                      fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
                      interpolation=cv2.INTER_CUBIC)

    # Bilateral filter
    gray = cv2.bilateralFilter(gray, BILATERAL_D, BILATERAL_SC, BILATERAL_SS)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    gray  = clahe.apply(gray)

    # Otsu binary
    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return original, [gray, otsu]


def detect_regions(original: np.ndarray, processed: np.ndarray) -> list:
    """
    Exact port of your detect_regions() function.
    Returns list with the single largest valid region dict.
    """
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    proc_h, proc_w = processed.shape[:2]
    orig_h, orig_w = original.shape[:2]
    scale_x    = orig_w / proc_w
    scale_y    = orig_h / proc_h
    image_area = proc_w * proc_h

    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < MIN_AREA or area > image_area * 0.9:
            continue
        if w < MIN_W or h < MIN_H:
            continue
        x1 = max(0, x - PAD);        y1 = max(0, y - PAD)
        x2 = min(proc_w, x + w + PAD); y2 = min(proc_h, y + h + PAD)
        crop_p = processed[y1:y2, x1:x2]
        ox1 = int(x1 * scale_x); oy1 = int(y1 * scale_y)
        ox2 = int(x2 * scale_x); oy2 = int(y2 * scale_y)
        crop_o = original[oy1:oy2, ox1:ox2]
        if crop_p.size == 0:
            continue
        regions.append({
            "processed": crop_p,
            "original":  crop_o,
            "bbox": {"x": ox1, "y": oy1,
                     "width": ox2 - ox1, "height": oy2 - oy1},
        })

    regions.sort(
        key=lambda r: r["bbox"]["width"] * r["bbox"]["height"],
        reverse=True,
    )
    if not regions:
        regions.append({
            "processed": processed,
            "original":  original,
            "bbox": {"x": 0, "y": 0, "width": orig_w, "height": orig_h},
        })
    return regions[:1]


def process_and_save(image_paths: list, output_dir: str) -> dict:
    """
    Process a list of images and save _gray.png + _otsu.png variants.
    Returns {original_path: {"gray": path, "otsu": path} or {"error": msg}}
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        try:
            original, (gray, otsu) = preprocess_image(img_path)
            gray_path = os.path.join(output_dir, f"{stem}_gray.png")
            otsu_path = os.path.join(output_dir, f"{stem}_otsu.png")
            cv2.imwrite(gray_path, gray)
            cv2.imwrite(otsu_path, otsu)
            results[img_path] = {"gray": gray_path, "otsu": otsu_path}
        except Exception as e:
            results[img_path] = {"error": str(e)}
    return results
