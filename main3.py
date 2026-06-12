"""
PaddleOCR Incremental Training System with Gemini Auto-Labeling
Version 7.0.0

5 Endpoints:
  POST /upload            - Upload image one by one
  GET  /status            - Full status with accuracy + total images trained
  GET  /status/{sid}      - Single session detail
  POST /extract           - OCR using your trained model
  POST /stop              - Stop stuck training, reset to idle

Changes in v7:
  1. TRAINING TIMEOUT — auto-kills training if it exceeds TRAIN_TIMEOUT_SECONDS (default 600s)
  2. LIVE LOG STREAMING — training stdout is captured line-by-line into session state
  3. /status and /status/{sid} now return last 50 training log lines so you can see exactly where it's stuck
  4. training_phase field tracks which step (cloning/labeling/training/export) is active
  5. All v6 fixes retained (no recursion, 1 label per image, /stop endpoint)
"""

import os
import re
import sys
import json
import time
import uuid
import signal
import random
import shutil
import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","AIzaSyATUrfdSLKxmP2_rs_kQdfG5az711atBoY")

BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
IMAGES_DIR    = DATA_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR    = DATA_DIR / "labels"
MODELS_DIR    = DATA_DIR / "models"
DICT_FILE     = DATA_DIR / "dict" / "digital_dict.txt"
SESSIONS_DIR  = DATA_DIR / "sessions"
WORK_DIR      = BASE_DIR / "work"
PADDLEOCR_DIR = WORK_DIR / "PaddleOCR"

NEW_IMAGES_PER_SESSION = 1    # change to 20 in production
OLD_IMAGES_SAMPLE      = 1    # change to 5 in production
TOTAL_BEFORE_TRAIN     = NEW_IMAGES_PER_SESSION + OLD_IMAGES_SAMPLE

CONFIDENCE_THRESHOLD   = 60.0
TRAIN_TIMEOUT_SECONDS  = 1800  # 30 min — enough for 50 epochs on CPU with a small dataset

for _d in [IMAGES_DIR, PROCESSED_DIR, LABELS_DIR,
           MODELS_DIR, DICT_FILE.parent, WORK_DIR, SESSIONS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")

app = FastAPI(title="PaddleOCR Incremental Trainer", version="7.0.0")

_ocr_cache = {"version": None, "predictor": None, "char_list": None}

# Global training process handle — used by /stop endpoint
_training_proc: Optional[asyncio.subprocess.Process] = None


# ─────────────────────────────────────────────────────────────────────────────
# OPENCV PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_and_save(img_path: Path) -> dict:
    """Grayscale → 4x upscale → bilateral → CLAHE → Otsu. Saves _gray + _otsu."""
    try:
        image = cv2.imread(str(img_path))
        if image is None:
            return {"error": f"Cannot load: {img_path.name}"}
        stem  = img_path.stem
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray  = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray  = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray_path = PROCESSED_DIR / f"{stem}_gray.png"
        otsu_path = PROCESSED_DIR / f"{stem}_otsu.png"
        cv2.imwrite(str(gray_path), gray)
        cv2.imwrite(str(otsu_path), otsu)
        return {"gray": str(gray_path), "otsu": str(otsu_path)}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _load_char_list(dict_path: Path) -> list:
    chars = ["blank"]
    if dict_path.exists():
        for line in dict_path.read_text(encoding="utf-8").splitlines():
            ch = line.rstrip("\n")
            if ch:
                chars.append(ch)
    else:
        chars += list("0123456789.,-+°CFV%AW/:")
    return chars


def _build_predictor(infer_dir: Path):
    import paddle.inference as paddle_infer
    pdmodel = infer_dir / "inference.pdmodel"
    json_m  = infer_dir / "inference.json"
    params  = infer_dir / "inference.pdiparams"
    model_file = str(pdmodel) if pdmodel.exists() else (
                 str(json_m)  if json_m.exists()  else None)
    if not model_file:
        raise FileNotFoundError(f"No inference model in {infer_dir}")
    if not params.exists():
        raise FileNotFoundError(f"inference.pdiparams not found in {infer_dir}")
    config = paddle_infer.Config(model_file, str(params))
    config.disable_gpu()
    config.disable_glog_info()
    return paddle_infer.create_predictor(config)


def _preprocess_for_rec(img_bgr: np.ndarray,
                         img_h: int = 48, img_w: int = 320) -> np.ndarray:
    h, w    = img_bgr.shape[:2]
    new_w   = min(int(w * img_h / h), img_w)
    resized = cv2.resize(img_bgr, (new_w, img_h))
    canvas  = np.zeros((img_h, img_w, 3), dtype=np.float32)
    canvas[:, :new_w, :] = resized.astype(np.float32)
    canvas  = (canvas / 255.0 - 0.5) / 0.5
    return canvas.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


def _ctc_decode(preds: np.ndarray, char_list: list) -> tuple:
    indices = np.argmax(preds, axis=1)
    scores  = np.max(preds, axis=1)
    chars, confs, prev = [], [], 0
    for i, idx in enumerate(indices):
        if idx != 0 and idx != prev:
            if idx < len(char_list):
                chars.append(char_list[idx])
                confs.append(float(scores[i]))
        prev = idx
    text = "".join(chars)
    conf = round(float(np.mean(confs)) * 100, 2) if confs else 0.0
    return text, conf


def _run_rec_on_image(img_bgr: np.ndarray, predictor, char_list: list) -> tuple:
    inp = _preprocess_for_rec(img_bgr)
    input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
    input_handle.reshape(inp.shape)
    input_handle.copy_from_cpu(inp)
    predictor.run()
    preds = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()
    return _ctc_decode(preds[0], char_list)


def find_latest_inference_dir() -> Optional[Path]:
    dirs = sorted(
        [p for p in MODELS_DIR.glob("v*/") if p.is_dir()],
        key=lambda p: int(p.name.replace("v","")) if p.name.replace("v","").isdigit() else 0
    )
    for vdir in reversed(dirs):
        infer = vdir / "inference"
        if (infer / "inference.pdiparams").exists() and (
            (infer / "inference.pdmodel").exists() or
            (infer / "inference.json").exists()
        ):
            return infer
    return None


def get_ocr_predictor():
    infer_dir = find_latest_inference_dir()
    if infer_dir is None:
        return None, None, None
    version = infer_dir.parent.name
    if _ocr_cache["version"] == version and _ocr_cache["predictor"] is not None:
        return _ocr_cache["predictor"], _ocr_cache["char_list"], version
    predictor = _build_predictor(infer_dir)
    char_list = _load_char_list(DICT_FILE)
    _ocr_cache.update({"version": version, "predictor": predictor, "char_list": char_list})
    print(f"[OCR] Loaded {version} — {len(char_list)} chars")
    return predictor, char_list, version


def _detect_regions(image: np.ndarray) -> list:
    gray_det = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    g = cv2.resize(gray_det, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    g = cv2.bilateralFilter(g, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    proc_h, proc_w = binary.shape[:2]
    orig_h, orig_w = image.shape[:2]
    scale_x, scale_y = orig_w / proc_w, orig_h / proc_h
    image_area = proc_w * proc_h
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 1000 or area > image_area * 0.9: continue
        if w < 80 or h < 40: continue
        pad = 20
        x1 = max(0, x-pad);       y1 = max(0, y-pad)
        x2 = min(proc_w, x+w+pad); y2 = min(proc_h, y+h+pad)
        ox1 = int(x1*scale_x); oy1 = int(y1*scale_y)
        ox2 = int(x2*scale_x); oy2 = int(y2*scale_y)
        crop = image[oy1:oy2, ox1:ox2]
        if crop.size == 0: continue
        regions.append({"bbox": {"x": ox1, "y": oy1,
                                  "width": ox2-ox1, "height": oy2-oy1},
                        "crop": crop})
    regions.sort(key=lambda r: r["bbox"]["width"] * r["bbox"]["height"], reverse=True)
    if not regions:
        regions.append({"bbox": {"x": 0, "y": 0, "width": orig_w, "height": orig_h},
                        "crop": image})
    return regions


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def _default_state() -> dict:
    return {
        "session_id": None, "status": "idle",
        "new_images": [], "sampled_old": [],
        "label_file": None, "model_version": None,
        "train_start": None, "train_end": None,
        "error": None, "total_labels_written": 0,
        "accuracy": None, "loss": None,
        "training_phase": None,        # current step: cloning/labeling/training/export
        "training_log": [],            # last 50 lines of training stdout (live)
    }


def get_session_file(sid: str) -> Path:
    return SESSIONS_DIR / f"{sid}.json"


def load_state(session_id: Optional[str] = None) -> dict:
    if session_id:
        sf = get_session_file(session_id)
        if sf.exists():
            try: return json.loads(sf.read_text())
            except Exception: pass
        return _default_state()
    files = sorted(SESSIONS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    if not files: return _default_state()
    try: return json.loads(files[0].read_text())
    except Exception: return _default_state()


def save_state(state: dict):
    sid = state.get("session_id")
    if sid:
        get_session_file(sid).write_text(json.dumps(state, indent=2, default=str))


def load_all_sessions() -> list:
    sessions = []
    for sf in sorted(SESSIONS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        try: sessions.append(json.loads(sf.read_text()))
        except Exception: continue
    return sessions


def next_version() -> str:
    existing = set()
    for p in list(MODELS_DIR.glob("v*/")) + list(LABELS_DIR.glob("label_v*.txt")):
        part = p.name.replace("label_","").replace(".txt","").replace("v","")
        try: existing.add(int(part))
        except ValueError: pass
    return f"v{max(existing, default=0) + 1}"


def find_previous_checkpoint() -> Optional[Path]:
    dirs = sorted(
        [p for p in MODELS_DIR.glob("v*/") if p.is_dir()],
        key=lambda p: int(p.name.replace("v","")) if p.name.replace("v","").isdigit() else 0
    )
    for vdir in reversed(dirs):
        best = vdir / "best_accuracy.pdparams"
        if best.exists():
            return vdir / "best_accuracy"
    return None


def get_old_images(exclude: list) -> list:
    exts = {".png",".jpg",".jpeg",".webp",".bmp"}
    return [p.name for p in IMAGES_DIR.glob("*")
            if p.suffix.lower() in exts
            and p.name not in exclude
            and "_crop" not in p.stem]  # skip auto-generated region crops


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI LABELING
# ─────────────────────────────────────────────────────────────────────────────

def gemini_extract_text(image_path: Path) -> str:
    """
    Single Gemini call — returns ALL text regions as a JSON array.
    Each element is a short string for one display region/line.
    Falls back to a plain string if JSON parsing fails.
    """
    import PIL.Image
    img = PIL.Image.open(image_path)
    prompt = (
        "This image shows a digital display, meter, or instrument panel. "
        "Extract every distinct text region (each number, value, or label). "
        "Return ONLY a JSON array of strings, one string per region, "
        "in reading order top-to-bottom left-to-right. "
        "Each string should contain only the exact characters visible "
        "(digits, decimal points, units like V, A, W, kWh, %, degC). "
        "No markdown, no code fences, no explanations. "
        "Example: [\"238.2V\", \"230.5V\", \"0.0V\", \"57.4kWh\"]"
    )
    response = gemini_model.generate_content([prompt, img])
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    try:
        items = json.loads(raw)
        if isinstance(items, list):
            # Clean each item
            return json.dumps([" ".join(str(t).split()) for t in items if str(t).strip()])
    except Exception:
        pass
    # Fallback: treat as plain text, return as single-element JSON array
    plain = " ".join(line.strip() for line in raw.splitlines() if line.strip())
    return json.dumps([plain]) if plain else json.dumps(["unknown"])


def label_image_by_regions(
    img_path: Path,
    label_file: Path,
    base_name: str,
) -> dict:
    """
    ONE Gemini call per image returns a JSON array of text regions.
    Each region is saved as a crop image + one label entry.
    Returns: {"crops": [...], "whole_text": str, "labels_written": int}
    """
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        return {"crops": [], "whole_text": "", "labels_written": 0, "error": "unreadable"}

    # ONE Gemini call → JSON list of region texts
    raw_json = gemini_extract_text(img_path)
    try:
        region_texts = json.loads(raw_json)
        if not isinstance(region_texts, list):
            region_texts = [str(region_texts)]
    except Exception:
        region_texts = [raw_json]

    region_texts = [t.strip() for t in region_texts if str(t).strip()]
    if not region_texts:
        region_texts = ["unknown"]

    # Detect image regions with OpenCV
    regions = _detect_regions(image_bgr)

    crops_info     = []
    labels_written = 0
    whole_text     = " ".join(region_texts)

    if regions:
        # Pair each detected region with a Gemini text (by index, cycle if needed)
        for i, region in enumerate(regions):
            crop = region["crop"]
            if len(crop.shape) == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

            crop_name = f"{base_name}_crop{i:02d}.png"
            crop_path = IMAGES_DIR / crop_name
            cv2.imwrite(str(crop_path), crop)

            # Use Gemini text at same index; cycle if fewer texts than regions
            text = region_texts[i % len(region_texts)]

            append_label(label_file, crop_name, text)
            labels_written += 1
            crops_info.append({"crop_name": crop_name, "text": text})
            print(f"[Label] {crop_name} → '{text}'")
    else:
        # No regions detected — label whole image with first text
        text = region_texts[0]
        append_label(label_file, img_path.name, text)
        labels_written = 1
        crops_info = [{"crop_name": img_path.name, "text": text}]
        print(f"[Label] No crops, whole image → '{text}'")

    return {
        "crops":          crops_info,
        "whole_text":     whole_text,
        "labels_written": labels_written,
    }

def append_label(label_file: Path, image_name: str, text: str):
    """
    Write one label line. Checks for duplicates — each image gets exactly ONE entry.
    """
    abs_img    = str((IMAGES_DIR / image_name).resolve())
    clean_text = " ".join(
        part.strip() for part in text.replace("\r","\n").split("\n") if part.strip()
    ) or "unknown"

    # Read existing entries and check for duplicate
    existing_paths = set()
    if label_file.exists():
        for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "\t" in line:
                existing_paths.add(line.split("\t", 1)[0].strip())

    if abs_img in existing_paths:
        print(f"[Label] Skipping duplicate: {image_name}")
        return  # already labeled — don't add again

    with open(label_file, "a", encoding="utf-8") as f:
        f.write(f"{abs_img}\t{clean_text}\n")


def find_label_in_history(image_name: str) -> Optional[str]:
    abs_img = str((IMAGES_DIR / image_name).resolve())
    for lf in sorted(LABELS_DIR.glob("label_v*.txt")):
        try:
            for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "\t" not in line: continue
                pp, tp = line.split("\t", 1)
                if pp.strip() == abs_img or pp.strip().endswith(image_name):
                    return tp.strip()
        except Exception: continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LABEL VALIDATION — stops training on invalid images instead of looping
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_fix_label_file(label_file: Path) -> dict:
    """Remove lines with no tab or missing image files."""
    if not label_file.exists():
        return {"good": 0, "removed": 0, "removed_lines": []}
    lines     = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    good, bad = [], []
    seen_paths = set()
    for line in lines:
        line = line.strip()
        if not line: continue
        if "\t" in line:
            pp, _ = line.split("\t", 1)
            pp = pp.strip()
            if not os.path.exists(pp):
                bad.append(f"missing_image:{line[:80]}")
            elif pp in seen_paths:
                bad.append(f"duplicate:{pp}")  # deduplicate here too
            else:
                good.append(line)
                seen_paths.add(pp)
        else:
            bad.append(f"no_tab:{line[:80]}")
    label_file.write_text("\n".join(good) + "\n" if good else "")
    return {"good": len(good), "removed": len(bad), "removed_lines": bad}


def pre_validate_images(label_file: Path) -> dict:
    """
    Verify every image is readable by OpenCV BEFORE training.
    Removes unreadable entries so training STOPS cleanly instead of
    hitting the RecursionError loop in simple_dataset.py.
    """
    if not label_file.exists():
        return {"readable": 0, "skipped": 0, "skipped_paths": []}
    lines   = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    good    = []
    skipped = []
    for line in lines:
        line = line.strip()
        if not line or "\t" not in line: continue
        path_part = line.split("\t", 1)[0].strip()
        img = cv2.imread(path_part)
        if img is None or img.size == 0:
            skipped.append(path_part)
            print(f"[Validate] INVALID image removed: {Path(path_part).name}")
        else:
            good.append(line)
    if skipped:
        label_file.write_text("\n".join(good) + "\n" if good else "")
        print(f"[Validate] Removed {len(skipped)} invalid images — training will stop if 0 remain")
    return {"readable": len(good), "skipped": len(skipped), "skipped_paths": skipped}


def prepare_training_label_file(label_file: Path) -> tuple:
    """
    Create a TEMP label file for PaddleOCR training.
    The temp file has entries duplicated to MIN 8 (prevents RecursionError).
    The original label file is NEVER modified — stays clean with 1 entry per image.
    Returns (temp_label_path, entry_count, unique_images_count)
    """
    if not label_file.exists():
        raise ValueError("Label file does not exist")

    unique_lines = [l.strip() for l in
                    label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if l.strip() and "\t" in l]
    unique_count = len(unique_lines)

    if unique_count == 0:
        raise ValueError("No valid entries in label file")

    # Pad to minimum 8 entries in the temp file only
    MIN_ENTRIES = 8
    train_lines = list(unique_lines)
    while len(train_lines) < MIN_ENTRIES:
        train_lines.extend(unique_lines)
    train_lines = train_lines[:max(MIN_ENTRIES, unique_count)]

    tmp_label = label_file.parent / f"_train_tmp_{label_file.stem}.txt"
    tmp_label.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    print(f"[Train] Temp label: {unique_count} unique images → {len(train_lines)} entries for training")

    return tmp_label, len(train_lines), unique_count


def patch_simple_dataset():
    """
    Fix the RecursionError WITHOUT touching simple_dataset.py.

    We write a tiny wrapper script (patch_guard.py) that is prepended to
    sys.path via PYTHONPATH when train.py runs.  Python imports OUR file
    first, which imports the real simple_dataset, then monkey-patches
    SimpleDataSet.__getitem__ at runtime — no file edits, no breakage.
    """
    guard_path = PADDLEOCR_DIR / "patch_guard"
    guard_path.mkdir(exist_ok=True)

    # This file lives at patch_guard/ppocr/data/simple_dataset.py
    # Python will import it INSTEAD of the real one, then we re-export
    # everything so the rest of PaddleOCR still works.
    pkg_dir = guard_path / "ppocr" / "data"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Write __init__.py files so it's a proper package
    (guard_path / "ppocr" / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    shim = guard_path / "ppocr" / "data" / "simple_dataset.py"
    shim.write_text(
        "# AUTO-GENERATED by main.py patch_simple_dataset() — DO NOT EDIT\n"
        "# Imports the real simple_dataset then patches __getitem__ to stop RecursionError\n"
        "import sys as _sys, os as _os, importlib as _importlib\n"
        "\n"
        "# Remove ourselves from sys.path so the real module loads next\n"
        "_guard_dir = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))\n"
        "if _guard_dir in _sys.path:\n"
        "    _sys.path.remove(_guard_dir)\n"
        "\n"
        "# Force-reload the real simple_dataset from PaddleOCR\n"
        "_key = 'ppocr.data.simple_dataset'\n"
        "if _key in _sys.modules:\n"
        "    del _sys.modules[_key]\n"
        "_real = _importlib.import_module('ppocr.data.simple_dataset')\n"
        "\n"
        "# Re-export EVERYTHING from the real module\n"
        "from ppocr.data.simple_dataset import *  # noqa\n"
        "try:\n"
        "    from ppocr.data.simple_dataset import SimpleDataSet, MultiScaleDataSet\n"
        "except ImportError:\n"
        "    pass\n"
        "\n"
        "# Monkey-patch __getitem__ with a recursion depth guard\n"
        "import functools as _ft\n"
        "def _safe_getitem(orig):\n"
        "    @_ft.wraps(orig)\n"
        "    def _wrapper(self, idx):\n"
        "        depth = getattr(self, '_pg_depth', 0)\n"
        "        if depth >= 3:\n"
        "            self._pg_depth = 0\n"
        "            return None\n"
        "        self._pg_depth = depth + 1\n"
        "        try:\n"
        "            return orig(self, idx)\n"
        "        finally:\n"
        "            self._pg_depth = max(0, getattr(self, '_pg_depth', 1) - 1)\n"
        "    return _wrapper\n"
        "\n"
        "try:\n"
        "    SimpleDataSet.__getitem__ = _safe_getitem(SimpleDataSet.__getitem__)\n"
        "    print('[PatchGuard] SimpleDataSet.__getitem__ protected against RecursionError')\n"
        "except Exception as _e:\n"
        "    print(f'[PatchGuard] Warning: {_e}')\n",
        encoding="utf-8"
    )
    print(f"[Patch] Guard shim written to {shim}")
    return str(guard_path)


# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY PARSING — reads accuracy from training log
# ─────────────────────────────────────────────────────────────────────────────

def parse_training_metrics(log_file: Path) -> dict:
    """
    Parse train.log to extract final accuracy and loss.
    Returns {"accuracy": float|None, "loss": float|None, "best_acc": float|None}
    """
    if not log_file.exists():
        return {"accuracy": None, "loss": None, "best_acc": None}

    text      = log_file.read_text(encoding="utf-8", errors="ignore")
    accuracy  = None
    loss      = None
    best_acc  = None

    # Parse last CTCLoss value
    loss_matches = re.findall(r"CTCLoss:\s*([\d.]+)", text)
    if loss_matches:
        try: loss = round(float(loss_matches[-1]), 4)
        except ValueError: pass

    # Parse accuracy from eval lines
    acc_matches = re.findall(r"acc:\s*([\d.]+)", text)
    if acc_matches:
        try:
            vals    = [float(v) for v in acc_matches]
            accuracy = round(vals[-1] * 100, 2) if vals[-1] <= 1.0 else round(vals[-1], 2)
            best_acc = round(max(vals) * 100, 2) if max(vals) <= 1.0 else round(max(vals), 2)
        except ValueError: pass

    # Parse "best metric, acc: X" line
    best_matches = re.findall(r"best metric,\s*acc:\s*([\d.]+)", text)
    if best_matches:
        try:
            v = float(best_matches[-1])
            best_acc = round(v * 100, 2) if v <= 1.0 else round(v, 2)
        except ValueError: pass

    return {"accuracy": accuracy, "loss": loss, "best_acc": best_acc}


def get_all_model_scores() -> list:
    """
    Return accuracy/loss for all trained versions by reading their train.log.
    """
    scores = []
    dirs   = sorted(
        [p for p in MODELS_DIR.glob("v*/") if p.is_dir()],
        key=lambda p: int(p.name.replace("v","")) if p.name.replace("v","").isdigit() else 0
    )
    for vdir in dirs:
        log_file   = vdir / "train.log"
        metrics    = parse_training_metrics(log_file)
        has_model  = (vdir / "best_accuracy.pdparams").exists()
        has_infer  = (vdir / "inference" / "inference.pdiparams").exists()
        scores.append({
            "version":       vdir.name,
            "has_model":     has_model,
            "has_inference": has_infer,
            "accuracy_pct":  metrics["accuracy"],
            "best_acc_pct":  metrics["best_acc"],
            "final_loss":    metrics["loss"],
        })
    return scores


def count_total_unique_trained_images() -> int:
    """
    Count unique image paths across ALL label files from successful sessions.
    This is the true count of images the model has been trained on.
    """
    all_paths = set()
    sessions  = load_all_sessions()
    for s in sessions:
        if s.get("status") != "done": continue
        lf = s.get("label_file")
        if not lf or not Path(lf).exists(): continue
        for line in Path(lf).read_text(encoding="utf-8", errors="ignore").splitlines():
            if "\t" in line:
                pp = line.split("\t", 1)[0].strip()
                if os.path.exists(pp):
                    all_paths.add(pp)
    return len(all_paths)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

async def run_training(state: dict):
    global _training_proc
    version    = state["model_version"]
    label_file = Path(state["label_file"])
    output_dir = MODELS_DIR / version

    state["status"]        = "training"
    state["train_start"]   = datetime.utcnow().isoformat()
    state["error"]         = None
    state["training_log"]  = []
    state["training_phase"] = "initializing"
    save_state(state)

    def _log(line: str, phase: str = None):
        """Append a line to the rolling in-session log (max 50 lines) and print it."""
        print(line)
        state["training_log"].append(line)
        if len(state["training_log"]) > 50:
            state["training_log"] = state["training_log"][-50:]
        if phase:
            state["training_phase"] = phase

    tmp_label = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: validate — remove bad/missing/duplicate entries
        _log("[Train] Step 1/7 — Validating label file...", phase="validating_labels")
        fix = validate_and_fix_label_file(label_file)
        _log(f"[Train] Validate: {fix['good']} good, {fix['removed']} removed")
        save_state(state)
        if fix["good"] == 0:
            raise ValueError("0 valid label entries. Check images exist and labels are correct.")

        # Step 2: pre-validate images are actually readable by OpenCV
        _log("[Train] Step 2/7 — Pre-validating image readability...", phase="validating_images")
        img_check = pre_validate_images(label_file)
        _log(f"[Train] Images: {img_check['readable']} readable, {img_check['skipped']} invalid removed")
        save_state(state)
        if img_check["readable"] == 0:
            raise ValueError(
                f"All {img_check['skipped']} images are unreadable/corrupt. "
                "Upload valid images before training."
            )

        # Step 3: create temp label file with padding (original stays clean — 1 per image)
        _log("[Train] Step 3/7 — Preparing temp label file...", phase="preparing_labels")
        tmp_label, train_count, unique_count = prepare_training_label_file(label_file)
        state["total_labels_written"] = unique_count
        _log(f"[Train] Temp label: {unique_count} unique → {train_count} entries (padded to min 8)")
        save_state(state)

        # Step 4: clone PaddleOCR if needed
        if not PADDLEOCR_DIR.exists():
            _log("[Train] Step 4/7 — Cloning PaddleOCR repo (first time only)...", phase="cloning_paddleocr")
            save_state(state)
            r = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth", "1",
                "https://github.com/PaddlePaddle/PaddleOCR.git",
                str(PADDLEOCR_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await r.communicate()
            if r.returncode != 0:
                raise RuntimeError(f"Clone failed: {out.decode()}")
            _log("[Train] PaddleOCR cloned successfully.")
        else:
            _log("[Train] Step 4/7 — PaddleOCR already present, skipping clone.")

        # Step 5: prepare recursion guard shim and restore simple_dataset if corrupted
        _log("[Train] Step 5/7 — Setting up recursion guard (PYTHONPATH shim)...", phase="patching")

        # Restore simple_dataset.py if a previous bad patch corrupted it
        ds_path = PADDLEOCR_DIR / "ppocr" / "data" / "simple_dataset.py"
        if ds_path.exists():
            ds_src = ds_path.read_text(encoding="utf-8")
            if "PATCHED_V7_RECURSION_GUARD" in ds_src or "AUTO-GENERATED by main.py" in ds_src:
                _log("[Patch] Detected corrupted/patched simple_dataset.py — restoring via git...")
                restore = await asyncio.create_subprocess_exec(
                    "git", "checkout", "ppocr/data/simple_dataset.py",
                    cwd=str(PADDLEOCR_DIR),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                )
                r_out, _ = await restore.communicate()
                if restore.returncode == 0:
                    _log("[Patch] simple_dataset.py restored to original.")
                else:
                    _log(f"[Patch] git restore warning: {r_out.decode(errors='replace')[:200]}")

        guard_dir = patch_simple_dataset()
        save_state(state)

        # Step 6: find previous checkpoint
        prev_ckpt       = find_previous_checkpoint()
        pretrained_args = ([f"Global.pretrained_model={prev_ckpt}"] if prev_ckpt else [])
        _log(f"[Train] Step 6/7 — Checkpoint: {prev_ckpt or 'none (fresh start)'}")

        label_abs  = str(tmp_label.resolve())   # use temp label (padded)
        images_abs = str(IMAGES_DIR.resolve())
        output_abs = str(output_dir.resolve())
        dict_abs   = str(DICT_FILE.resolve())

        # Step 7: run training with TIMEOUT
        train_cmd = [
            sys.executable, "tools/train.py",
            "-c", "configs/rec/PP-OCRv3/en_PP-OCRv3_mobile_rec.yml",
            "-o",
            f"Train.dataset.data_dir={images_abs}",
            f"Train.dataset.label_file_list=['{label_abs}']",
            f"Eval.dataset.data_dir={images_abs}",
            f"Eval.dataset.label_file_list=['{label_abs}']",
            f"Global.save_model_dir={output_abs}",
            f"Global.character_dict_path={dict_abs}",
            "Global.use_gpu=False",
            "Global.epoch_num=10",            # enough epochs for tiny datasets
            "Global.save_epoch_step=10",
            "Global.eval_batch_step=[0,8]",   # evaluate every 8 iters (~1 epoch with 8 padded samples)
            "Global.cal_metric_during_train=True",
            "Global.log_smooth_window=1",
            "Train.loader.num_workers=0",
            "Train.loader.batch_size_per_card=1",
            "Train.loader.drop_last=False",
            "Eval.loader.num_workers=0",
            "Eval.loader.batch_size_per_card=1",
            "Eval.loader.drop_last=False",
        ] + pretrained_args

        _log(f"[Train] Step 7/7 — Launching PaddleOCR train.py "
             f"({unique_count} images, 3 epochs, timeout={TRAIN_TIMEOUT_SECONDS}s)...",
             phase="training")
        save_state(state)

        # Inject guard_dir at front of PYTHONPATH so our shim loads first
        train_env = os.environ.copy()
        existing_pypath = train_env.get("PYTHONPATH", "")
        train_env["PYTHONPATH"] = (
            guard_dir + os.pathsep + existing_pypath
            if existing_pypath else guard_dir
        )
        _log(f"[Train] PYTHONPATH guard: {guard_dir}")

        _training_proc = await asyncio.create_subprocess_exec(
            *train_cmd, cwd=str(PADDLEOCR_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=train_env,
        )

        # ── Stream stdout line-by-line with a per-read timeout ──────────────
        train_start_ts = time.time()
        rc = None
        try:
            while True:
                elapsed = time.time() - train_start_ts
                remaining = TRAIN_TIMEOUT_SECONDS - elapsed
                if remaining <= 0:
                    _log(f"[Train] TIMEOUT — exceeded {TRAIN_TIMEOUT_SECONDS}s. Killing process.")
                    try:
                        _training_proc.terminate()
                        await asyncio.sleep(2)
                        _training_proc.kill()
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Training timed out after {TRAIN_TIMEOUT_SECONDS}s. "
                        "Increase TRAIN_TIMEOUT_SECONDS or reduce epoch_num."
                    )
                try:
                    line_bytes = await asyncio.wait_for(
                        _training_proc.stdout.readline(),
                        timeout=min(30.0, remaining)   # re-check timeout every 30s at most
                    )
                except asyncio.TimeoutError:
                    # No output for 30s — log a heartbeat so /status shows it's alive
                    elapsed = int(time.time() - train_start_ts)
                    _log(f"[Train] ... still running ({elapsed}s elapsed, no output in last 30s)")
                    save_state(state)
                    await asyncio.sleep(0)
                    continue

                if not line_bytes:
                    # EOF — process finished
                    break
                line = line_bytes.decode(errors="replace").rstrip()
                _log(line)
                # Save state after every line so /status always has fresh data
                save_state(state)
                # Yield event loop so FastAPI can serve /status requests mid-training
                await asyncio.sleep(0)

            rc = await _training_proc.wait()
        finally:
            _training_proc = None

        if rc is not None and rc != 0 and rc != -15:
            recent = "\n".join(state["training_log"][-20:])
            raise RuntimeError(f"train.py exited with code {rc}. Last output:\n{recent}")

        if rc == -15:
            raise RuntimeError("Training was stopped manually via /stop endpoint.")

        # Step 8: save best_accuracy.pdparams
        _log("[Train] Saving best checkpoint...", phase="saving_checkpoint")
        saved = sorted(output_dir.glob("*.pdparams"), key=os.path.getmtime)
        if not saved:
            raise FileNotFoundError(f"No .pdparams in {output_dir}")
        best = output_dir / "best_accuracy.pdparams"
        if not best.exists():
            shutil.copy2(saved[-1], best)
            _log(f"[Train] best_accuracy saved from {saved[-1].name}")

        # Cleanup
        for f in output_dir.glob("*.pdparams"):
            if f.stem != "best_accuracy": f.unlink(missing_ok=True)
        for f in output_dir.glob("*.pdopt"):
            f.unlink(missing_ok=True)

        # Parse accuracy from log
        log_file = output_dir / "train.log"
        metrics  = parse_training_metrics(log_file)
        state["accuracy"] = metrics.get("best_acc") or metrics.get("accuracy")
        state["loss"]     = metrics.get("loss")
        _log(f"[Train] Accuracy: {state['accuracy']}%  Loss: {state['loss']}")

        # Step 9: export inference model
        _log("[Train] Exporting inference model...", phase="exporting")
        save_state(state)
        infer_dir = output_dir / "inference"
        infer_dir.mkdir(exist_ok=True)
        export_cmd = [
            sys.executable, "tools/export_model.py",
            "-c", "configs/rec/PP-OCRv3/en_PP-OCRv3_mobile_rec.yml",
            "-o",
            f"Global.pretrained_model={str(output_dir / 'best_accuracy')}",
            f"Global.save_inference_dir={str(infer_dir)}",
            f"Global.character_dict_path={dict_abs}",
        ]
        exp = await asyncio.create_subprocess_exec(
            *export_cmd, cwd=str(PADDLEOCR_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        exp_out, _ = await exp.communicate()
        if exp.returncode != 0:
            _log(f"[Train] Export warning: {exp_out.decode(errors='replace')[-300:]}")
        else:
            _ocr_cache["version"] = None  # reload predictor
            _log(f"[Train] Inference exported: {infer_dir}")

        # Keep only 2 latest model dirs
        all_vdirs = sorted(
            [p for p in MODELS_DIR.glob("v*/") if p.is_dir()],
            key=lambda p: int(p.name.replace("v","")) if p.name.replace("v","").isdigit() else 0
        )
        for old in all_vdirs[:-2]:
            shutil.rmtree(old, ignore_errors=True)
            _log(f"[Cleanup] Deleted old model dir: {old.name}")

        (MODELS_DIR / f"{version}.pd").touch()
        state["status"]         = "done"
        state["train_end"]      = datetime.utcnow().isoformat()
        state["error"]          = None
        state["training_phase"] = "complete"
        _log(f"[Train] ✓ Complete: {version}")

    except Exception as e:
        state["status"]         = "error"
        state["error"]          = str(e)
        state["training_phase"] = "failed"
        _log(f"[Train] ERROR: {e}")

    finally:
        # Always clean up temp label file
        if tmp_label and tmp_label.exists():
            tmp_label.unlink(missing_ok=True)
        _training_proc = None

    save_state(state)


async def prepare_and_train(state: dict):
    old_available = get_old_images(exclude=state["new_images"])
    sampled       = random.sample(old_available,
                                  min(OLD_IMAGES_SAMPLE, len(old_available)))
    state["sampled_old"] = sampled
    label_file = Path(state["label_file"])

    for img_name in sampled:
        img_path  = IMAGES_DIR / img_name
        gray_path = PROCESSED_DIR / f"{img_path.stem}_gray.png"
        otsu_path = PROCESSED_DIR / f"{img_path.stem}_otsu.png"
        if not gray_path.exists() or not otsu_path.exists():
            res = preprocess_and_save(img_path)
            print(f"[Preprocess] {'OK' if 'error' not in res else 'FAIL'}: {img_name}")

        # Check if this old image already has region-crop labels in history
        # (crop filenames follow the pattern stem_crop00.png, stem_crop01.png …)
        crop_pattern = f"{img_path.stem}_crop"
        already_labeled = any(
            crop_pattern in line
            for lf in sorted(LABELS_DIR.glob("label_v*.txt"))
            for line in (lf.read_text(encoding="utf-8", errors="ignore").splitlines()
                         if lf.exists() else [])
            if "	" in line
        )

        if already_labeled:
            # Re-use existing crop labels by appending them to this label file
            for lf in sorted(LABELS_DIR.glob("label_v*.txt")):
                if not lf.exists(): continue
                for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if "	" not in line: continue
                    pp, tt = line.split("	", 1)
                    if crop_pattern in pp:
                        crop_name = Path(pp.strip()).name
                        append_label(label_file, crop_name, tt.strip())
            print(f"[Label] Re-used existing crop labels for {img_name}")
        else:
            # No prior crop labels — run region labeling now
            try:
                result = label_image_by_regions(img_path, label_file, base_name=img_path.stem)
                print(f"[Label] {img_name} → {result['labels_written']} crops labeled")
            except Exception as e:
                print(f"[Label] Region labeling failed for {img_name}: {e}")
                # Fallback: whole-image label
                text = find_label_in_history(img_name) or "unknown"
                append_label(label_file, img_name, text)

    save_state(state)
    await run_training(state)


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

def backfill_processed_images():
    exts     = {".jpg",".jpeg",".png",".bmp",".webp"}
    all_imgs = [p for p in IMAGES_DIR.glob("*") if p.suffix.lower() in exts]
    missing  = [p for p in all_imgs
                if not (PROCESSED_DIR / f"{p.stem}_gray.png").exists()
                or not (PROCESSED_DIR / f"{p.stem}_otsu.png").exists()]
    if missing:
        print(f"[Startup] Backfilling {len(missing)} unprocessed images...")
        for img in missing:
            r = preprocess_and_save(img)
            print(f"  {'OK' if 'error' not in r else 'FAIL'}: {img.name}")
    else:
        print(f"[Startup] All {len(all_imgs)} images already processed.")


backfill_processed_images()

# Pre-build the recursion-guard shim if PaddleOCR is already cloned
if PADDLEOCR_DIR.exists():
    patch_simple_dataset()
    print("[Startup] Recursion guard shim ready.")
else:
    print("[Startup] PaddleOCR not cloned yet — shim will be built before first training.")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    state = load_state()

    if state["status"] == "training":
        raise HTTPException(status_code=429, detail={
            "error":      "Training in progress. Wait or call POST /stop to cancel.",
            "status":     state["status"],
            "session_id": state["session_id"],
        })

    if (state["status"] == "collecting"
            and len(state["new_images"]) >= NEW_IMAGES_PER_SESSION):
        raise HTTPException(status_code=429, detail={
            "error":  f"Already have {NEW_IMAGES_PER_SESSION} images. Training starting soon.",
            "status": state["status"],
        })

    if state["status"] in ("idle", "done", "error"):
        version    = next_version()
        session_id = f"session_{version}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        state      = _default_state()
        label_path = (LABELS_DIR / f"label_{version}.txt").resolve()
        state.update({
            "session_id":    session_id,
            "status":        "collecting",
            "model_version": version,
            "label_file":    str(label_path),
        })
        label_path.touch()
        save_state(state)

    suffix   = Path(file.filename or "image.jpg").suffix.lower() or ".jpg"
    img_name = f"{state['session_id']}_{len(state['new_images']):03d}{suffix}"
    img_path = IMAGES_DIR / img_name
    img_path.write_bytes(await file.read())

    proc_result = preprocess_and_save(img_path)

    # Label by detected regions (one label per crop = short strings = real accuracy)
    try:
        label_result = label_image_by_regions(
            img_path,
            Path(state["label_file"]),
            base_name=img_path.stem,
        )
    except Exception as e:
        img_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Labeling failed: {e}")

    state["new_images"].append(img_name)
    state["total_labels_written"] = (
        state.get("total_labels_written", 0) + label_result["labels_written"]
    )
    save_state(state)

    collected = len(state["new_images"])
    response  = {
        "session_id":        state["session_id"],
        "model_version":     state["model_version"],
        "image_saved":       img_name,
        "extracted_text":    label_result["whole_text"],
        "region_crops":      label_result["crops"],        # per-crop labels
        "labels_written":    label_result["labels_written"],
        "processed": {
            "gray":  proc_result.get("gray"),
            "otsu":  proc_result.get("otsu"),
            "error": proc_result.get("error"),
        },
        "new_images_count": collected,
        "target":           NEW_IMAGES_PER_SESSION,
        "remaining":        max(0, NEW_IMAGES_PER_SESSION - collected),
        "status":           state["status"],
        "message":          f"Image {collected}/{NEW_IMAGES_PER_SESSION} collected. "
                            f"{label_result['labels_written']} crop labels created.",
    }

    if collected >= NEW_IMAGES_PER_SESSION:
        response["message"] = (
            f"Threshold reached! Sampling {OLD_IMAGES_SAMPLE} old images "
            "and starting training automatically."
        )
        background_tasks.add_task(prepare_and_train, state)

    return JSONResponse(content=response)


@app.get("/status")
async def get_status():
    """
    Full status — always reads fresh from disk, responds immediately even during training.
    """
    loop = asyncio.get_event_loop()
    sessions      = await loop.run_in_executor(None, load_all_sessions)
    model_scores  = await loop.run_in_executor(None, get_all_model_scores)
    total_trained = await loop.run_in_executor(None, count_total_unique_trained_images)
    label_files    = sorted([p.name for p in LABELS_DIR.glob("label_v*.txt")])
    total_on_disk  = len([p for p in IMAGES_DIR.glob("*")
                          if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}])
    total_proc     = len(list(PROCESSED_DIR.glob("*_gray.png")))
    prev_ckpt      = find_previous_checkpoint()
    infer_ready    = find_latest_inference_dir()

    trained_versions = [m["version"] for m in model_scores if m["has_model"]]

    # Current session summary
    current = load_state()

    return {
        "summary": {
            "total_images_trained":    total_trained,
            "total_images_on_disk":    total_on_disk,
            "total_processed_images":  total_proc,
            "total_sessions":          len(sessions),
            "trained_versions_count":  len(trained_versions),
            "latest_inference_model":  infer_ready.parent.name if infer_ready else None,
            "current_status":          current.get("status", "idle"),
        },
        "model_scores": model_scores,
        "current_session": {
            "session_id":           current.get("session_id"),
            "status":               current.get("status", "idle"),
            "model_version":        current.get("model_version"),
            "new_images_count":     len(current.get("new_images", [])),
            "sampled_old_count":    len(current.get("sampled_old", [])),
            "total_labels_written": current.get("total_labels_written", 0),
            "label_file":           current.get("label_file"),
            "train_start":          current.get("train_start"),
            "train_end":            current.get("train_end"),
            "accuracy_pct":         current.get("accuracy"),
            "final_loss":           current.get("loss"),
            "error":                current.get("error"),
            "training_phase":       current.get("training_phase"),
            "training_log":         current.get("training_log", []),
        },
        "incremental_chain": {
            "trained_versions": trained_versions,
            "next_version":     f"v{len(trained_versions)+1}",
            "prev_checkpoint":  str(prev_ckpt)+".pdparams" if prev_ckpt else None,
            "chain":            " -> ".join(trained_versions + [f"v{len(trained_versions)+1}(next)"]),
        },
        "storage": {
            "label_files":   label_files,
            "models_dir":    str(MODELS_DIR),
            "processed_dir": str(PROCESSED_DIR),
        },
        "thresholds": {
            "new_images_per_session": NEW_IMAGES_PER_SESSION,
            "old_images_sampled":     OLD_IMAGES_SAMPLE,
            "total_before_training":  TOTAL_BEFORE_TRAIN,
        },
        "sessions": sessions,
    }


@app.get("/status/{session_id}")
async def get_session_status(session_id: str):
    state     = load_state(session_id)
    collected = len(state["new_images"])
    return {
        "session_id":           state["session_id"],
        "status":               state["status"],
        "model_version":        state["model_version"],
        "label_file":           state["label_file"],
        "new_images":           state["new_images"],
        "new_images_count":     collected,
        "sampled_old_images":   state["sampled_old"],
        "sampled_old_count":    len(state["sampled_old"]),
        "total_labels_written": state["total_labels_written"],
        "train_start":          state["train_start"],
        "train_end":            state["train_end"],
        "accuracy_pct":         state.get("accuracy"),
        "final_loss":           state.get("loss"),
        "error":                state["error"],
        "training_phase":       state.get("training_phase"),
        "training_log":         state.get("training_log", []),
        "progress": {
            "phase": (
                "training"      if state["status"] == "training"
                else "waiting"  if collected < NEW_IMAGES_PER_SESSION
                else "starting_soon"
            ),
            "images_collected":  collected,
            "images_needed":     NEW_IMAGES_PER_SESSION,
            "images_remaining":  max(0, NEW_IMAGES_PER_SESSION - collected),
            "percent_collected": round(collected / NEW_IMAGES_PER_SESSION * 100, 1),
        },
    }


@app.post("/stop")
async def stop_training():
    """
    Stop any running training process and reset state to idle.
    Use this when training is stuck — then upload new images to start fresh.
    """
    global _training_proc

    stopped_pid = None

    # Kill the asyncio subprocess if we have a handle
    if _training_proc is not None:
        try:
            _training_proc.terminate()
            stopped_pid = _training_proc.pid
            print(f"[Stop] Sent SIGTERM to training process PID {stopped_pid}")
            await asyncio.sleep(2)
            try:
                _training_proc.kill()  # force kill if still running
            except Exception:
                pass
        except Exception as e:
            print(f"[Stop] Error terminating process: {e}")
        _training_proc = None

    # Also kill any orphaned train.py processes
    try:
        result = subprocess.run(
            ["pkill", "-u", os.environ.get("USER", ""), "-f", "train.py"],
            capture_output=True
        )
        print(f"[Stop] pkill train.py: {result.returncode}")
    except Exception as e:
        print(f"[Stop] pkill failed: {e}")

    # Reset stuck sessions
    reset_count = 0
    for sf in SESSIONS_DIR.glob("*.json"):
        try:
            s = json.loads(sf.read_text())
            if s.get("status") in ("training", "collecting"):
                s["status"]    = "error"
                s["error"]     = "Stopped manually via POST /stop"
                s["train_end"] = datetime.utcnow().isoformat()
                sf.write_text(json.dumps(s, indent=2))
                reset_count += 1
                print(f"[Stop] Reset session: {s['session_id']}")
        except Exception:
            continue

    # Clean up temp label files
    for tmp in LABELS_DIR.glob("_train_tmp_*.txt"):
        tmp.unlink(missing_ok=True)

    # Rebuild the recursion guard shim so next training run is clean
    if PADDLEOCR_DIR.exists():
        # Restore simple_dataset.py if it was corrupted by an old patch
        ds_path = PADDLEOCR_DIR / "ppocr" / "data" / "simple_dataset.py"
        if ds_path.exists():
            ds_src = ds_path.read_text(encoding="utf-8")
            if "PATCHED_V7_RECURSION_GUARD" in ds_src or "AUTO-GENERATED by main.py" in ds_src:
                import subprocess as _sp
                _sp.run(["git", "checkout", "ppocr/data/simple_dataset.py"],
                        cwd=str(PADDLEOCR_DIR), capture_output=True)
                print("[Stop] simple_dataset.py restored to original via git")
        patch_simple_dataset()  # Rebuild clean shim

    return {
        "message":       "Training stopped. System reset to idle. You can now upload new images.",
        "stopped_pid":   stopped_pid,
        "sessions_reset": reset_count,
        "next_action":   "POST /upload to start a new training session",
    }


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    tablet_id: str = "default",
):
    t_start    = time.time()
    request_id = str(uuid.uuid4())

    predictor, char_list, model_version = get_ocr_predictor()
    queue_wait_ms = round((time.time() - t_start) * 1000, 2)

    if predictor is None:
        raise HTTPException(status_code=503, detail=(
            "No trained model available yet. "
            "Upload images to trigger training first."
        ))

    suffix   = Path(file.filename or "img.jpg").suffix.lower() or ".jpg"
    tmp_path = DATA_DIR / f"_extract_{request_id}{suffix}"
    try:
        tmp_path.write_bytes(await file.read())
        image_bgr = cv2.imread(str(tmp_path))
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Cannot read image file.")

        regions           = _detect_regions(image_bgr)
        readings          = []
        low_conf_readings = []
        raw_texts         = []

        for region in regions:
            crop = region["crop"]
            bbox = region["bbox"]
            if len(crop.shape) == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            text, confidence = _run_rec_on_image(crop, predictor, char_list)
            if not text: continue
            raw_texts.append(text)
            entry = {"value": text, "unit": "", "confidence": confidence, "bbox": bbox}
            if confidence >= CONFIDENCE_THRESHOLD:
                readings.append(entry)
            else:
                low_conf_readings.append({
                    "value":      text,
                    "confidence": confidence,
                    "warning":    (f"Confidence {confidence:.1f}% below "
                                  f"threshold {CONFIDENCE_THRESHOLD:.1f}%"),
                })

        return {
            "request_id":              request_id,
            "tablet_id":               tablet_id,
            "image_name":              file.filename,
            "readings":                readings,
            "low_confidence_readings": low_conf_readings,
            "raw_text":                "\n".join(raw_texts),
            "processing_time_ms":      round((time.time() - t_start) * 1000, 2),
            "model_used":              f"{model_version} (custom trained)",
            "queue_wait_ms":           queue_wait_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
