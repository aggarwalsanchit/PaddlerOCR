"""
PaddleOCR Incremental Training System with Gemini Auto-Labeling
4 Endpoints:
  POST /upload        - Upload image one by one (Gemini labels it automatically)
  GET  /status        - Overall system status + history
  GET  /status/{sid}  - Single session detail
  POST /extract       - OCR using your trained model with detailed response

Fixes in this version:
  - drop_last=False  → prevents empty batch RecursionError
  - MIN_ENTRIES pad  → duplicates tiny datasets to min 8 entries
  - Image pre-validation → skips unreadable images before training
  - simple_dataset patch → max 3 retries instead of 982
  - Full Gemini text extraction (all lines preserved)
"""

import os
import sys
import json
import time
import uuid
import random
import shutil
import asyncio
import tempfile
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

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
MIN_LABEL_ENTRIES      = 8    # pad dataset to this minimum to prevent RecursionError

CONFIDENCE_THRESHOLD   = 60.0

for _d in [IMAGES_DIR, PROCESSED_DIR, LABELS_DIR,
           MODELS_DIR, DICT_FILE.parent, WORK_DIR, SESSIONS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")

app = FastAPI(title="PaddleOCR Incremental Trainer", version="5.0.0")

_ocr_cache = {"version": None, "predictor": None, "char_list": None}


# ─────────────────────────────────────────────────────────────────────────────
# OPENCV PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_and_save(img_path: Path) -> dict:
    """
    Exact pipeline: grayscale -> 4x upscale -> bilateral -> CLAHE -> Otsu.
    Saves _gray.png and _otsu.png to data/processed/.
    """
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
# CUSTOM INFERENCE ENGINE  (bypasses PaddleOCR 3.x API entirely)
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
    if pdmodel.exists():
        model_file = str(pdmodel)
    elif json_m.exists():
        model_file = str(json_m)
    else:
        raise FileNotFoundError(f"No inference model found in {infer_dir}")
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
    indices   = np.argmax(preds, axis=1)
    scores    = np.max(preds, axis=1)
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
    version_dirs = sorted(
        [p for p in MODELS_DIR.glob("v*/") if p.is_dir()],
        key=lambda p: int(p.name.replace("v","")) if p.name.replace("v","").isdigit() else 0
    )
    for vdir in reversed(version_dirs):
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
    print(f"[OCR] Loading: {infer_dir}")
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
        x1 = max(0, x-pad);      y1 = max(0, y-pad)
        x2 = min(proc_w, x+w+pad); y2 = min(proc_h, y+h+pad)
        ox1 = int(x1*scale_x);   oy1 = int(y1*scale_y)
        ox2 = int(x2*scale_x);   oy2 = int(y2*scale_y)
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
    }


def get_session_file(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def load_state(session_id: Optional[str] = None) -> dict:
    if session_id:
        sf = get_session_file(session_id)
        if sf.exists():
            try: return json.loads(sf.read_text())
            except Exception: pass
        return _default_state()
    session_files = sorted(SESSIONS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    if not session_files: return _default_state()
    try: return json.loads(session_files[0].read_text())
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
    version_dirs = sorted(
        [p for p in MODELS_DIR.glob("v*/") if p.is_dir()],
        key=lambda p: int(p.name.replace("v","")) if p.name.replace("v","").isdigit() else 0
    )
    for vdir in reversed(version_dirs):
        best = vdir / "best_accuracy.pdparams"
        if best.exists():
            return vdir / "best_accuracy"
    return None


def get_old_images(exclude: list) -> list:
    exts = {".png",".jpg",".jpeg",".webp",".bmp"}
    return [p.name for p in IMAGES_DIR.glob("*")
            if p.suffix.lower() in exts and p.name not in exclude]


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI LABELING
# ─────────────────────────────────────────────────────────────────────────────

def gemini_extract_text(image_path: Path) -> str:
    """
    Extract ALL visible text from image — all lines preserved and merged
    into one training label line (PaddleOCR rec model expects single-line labels).
    """
    import PIL.Image
    img    = PIL.Image.open(image_path)
    prompt = """Extract ALL visible text from this image exactly as written.

Requirements:
- Return EVERY readable word, number, symbol
- Include all lines from top to bottom
- Preserve numbers, units, symbols exactly
- NO markdown, NO bullet points, NO explanations
- Return plain text only

Example output:
EA DIP205G-4NLED 4x20 LED backlight max. 150mA @ 4.2V 3.3V or 5V supply"""

    response   = gemini_model.generate_content([prompt, img])
    raw        = response.text.strip()

    # Remove markdown bullets
    raw = raw.replace("* ", "").replace("- ", "")

    # Merge all lines into one training label
    # PaddleOCR rec model expects single-line labels
    merged = " ".join(
        line.strip()
        for line in raw.splitlines()
        if line.strip()
    )
    return merged


def append_label(label_file: Path, image_name: str, text: str):
    """Write single-line PaddleOCR label: /abs/path/img.jpg\ttext"""
    abs_img    = str((IMAGES_DIR / image_name).resolve())
    clean_text = " ".join(
        part.strip() for part in text.replace("\r","\n").split("\n") if part.strip()
    ) or "unknown"
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


def validate_and_fix_label_file(label_file: Path) -> dict:
    """Remove lines with no tab or missing image files."""
    if not label_file.exists():
        return {"good": 0, "removed": 0, "removed_lines": []}
    lines     = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    good, bad = [], []
    for line in lines:
        line = line.strip()
        if not line: continue
        if "\t" in line:
            pp, _ = line.split("\t", 1)
            if os.path.exists(pp.strip()):
                good.append(line)
            else:
                bad.append(f"missing_image:{line[:80]}")
        else:
            bad.append(f"no_tab:{line[:80]}")
    label_file.write_text("\n".join(good) + "\n" if good else "")
    return {"good": len(good), "removed": len(bad), "removed_lines": bad}


def pre_validate_images(label_file: Path) -> dict:
    """
    Read every image path in the label file and verify it is
    actually loadable by OpenCV. Remove unreadable entries.
    Prevents simple_dataset RecursionError on corrupt images.
    """
    if not label_file.exists():
        return {"readable": 0, "skipped": 0, "skipped_paths": []}

    lines    = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    good     = []
    skipped  = []

    for line in lines:
        line = line.strip()
        if not line or "\t" not in line:
            continue
        path_part = line.split("\t", 1)[0].strip()
        img = cv2.imread(path_part)
        if img is None or img.size == 0:
            skipped.append(path_part)
            print(f"[Validate] Skipping unreadable: {Path(path_part).name}")
        else:
            good.append(line)

    if skipped:
        label_file.write_text("\n".join(good) + "\n" if good else "")
        print(f"[Validate] Removed {len(skipped)} unreadable images")

    return {"readable": len(good), "skipped": len(skipped), "skipped_paths": skipped}


def pad_label_file(label_file: Path, min_entries: int = MIN_LABEL_ENTRIES) -> int:
    """
    Duplicate entries until label file has at least min_entries.
    Prevents PaddleOCR simple_dataset RecursionError on tiny datasets.
    Returns final entry count.
    """
    lines = [l for l in label_file.read_text(
        encoding="utf-8", errors="ignore").splitlines() if l.strip()]

    if len(lines) >= min_entries:
        return len(lines)

    original = len(lines)
    while len(lines) < min_entries:
        lines.extend(lines)
    lines = lines[:min_entries]
    label_file.write_text("\n".join(lines) + "\n")
    print(f"[Train] Padded label file: {original} -> {len(lines)} entries "
          f"(min {min_entries} required)")
    return len(lines)


def patch_simple_dataset():
    """
    Patch PaddleOCR's simple_dataset.py to stop retrying after 3 attempts
    instead of recursing 982 times on bad images.
    One-time patch, safe to call multiple times.
    """
    ds_path = PADDLEOCR_DIR / "ppocr" / "data" / "simple_dataset.py"
    if not ds_path.exists():
        return  # repo not cloned yet, will patch after clone

    src = ds_path.read_text(encoding="utf-8")

    # Already patched?
    if "_retry_depth" in src:
        return

    old = "            rnd_idx = random.randint(0, self.__len__() - 1)\n                return self.__getitem__(rnd_idx)"
    new = ("            # Max 3 retries — prevents RecursionError on tiny/corrupt datasets\n"
           "            _depth = getattr(self, '_retry_depth', 0)\n"
           "            if _depth >= 3:\n"
           "                self._retry_depth = 0\n"
           "                return None\n"
           "            self._retry_depth = _depth + 1\n"
           "            rnd_idx = random.randint(0, self.__len__() - 1)\n"
           "            result = self.__getitem__(rnd_idx)\n"
           "            self._retry_depth = 0\n"
           "            return result")

    if old in src:
        ds_path.write_text(src.replace(old, new), encoding="utf-8")
        print("[Patch] simple_dataset.py patched — max 3 retries on bad images")
    else:
        print("[Patch] simple_dataset.py pattern not found — skipping patch "
              "(may differ in this PaddleOCR version)")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING  (v1 -> v2 -> v3 incremental chain)
# ─────────────────────────────────────────────────────────────────────────────

async def run_training(state: dict):
    version    = state["model_version"]
    label_file = Path(state["label_file"])
    output_dir = MODELS_DIR / version

    state["status"]      = "training"
    state["train_start"] = datetime.utcnow().isoformat()
    state["error"]       = None
    save_state(state)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: validate label file ───────────────────────────────────────
        fix = validate_and_fix_label_file(label_file)
        print(f"[Train] Labels: {fix['good']} good, {fix['removed']} removed")
        if fix["good"] == 0:
            raise ValueError("0 valid label entries after cleanup.")

        # ── Step 2: pre-validate images are actually readable ─────────────────
        img_check = pre_validate_images(label_file)
        print(f"[Train] Images: {img_check['readable']} readable, "
              f"{img_check['skipped']} skipped")
        if img_check["readable"] == 0:
            raise ValueError(
                f"All images in label file are unreadable. "
                f"Skipped: {img_check['skipped_paths']}"
            )

        # ── Step 3: pad to minimum entries ────────────────────────────────────
        final_count = pad_label_file(label_file, MIN_LABEL_ENTRIES)
        print(f"[Train] Final label entries for training: {final_count}")

        # ── Step 4: clone PaddleOCR if needed ────────────────────────────────
        if not PADDLEOCR_DIR.exists():
            print("[Train] Cloning PaddleOCR...")
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

        # ── Step 5: patch simple_dataset.py ──────────────────────────────────
        patch_simple_dataset()

        # ── Step 6: find previous checkpoint ─────────────────────────────────
        prev_ckpt       = find_previous_checkpoint()
        pretrained_args = ([f"Global.pretrained_model={prev_ckpt}"]
                           if prev_ckpt else [])
        print(f"[Train] Prev checkpoint: {prev_ckpt or 'none (fresh start)'}")

        label_abs  = str(label_file.resolve())
        images_abs = str(IMAGES_DIR.resolve())
        output_abs = str(output_dir.resolve())
        dict_abs   = str(DICT_FILE.resolve())

        # ── Step 7: run training ──────────────────────────────────────────────
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
            "Global.epoch_num=3",
            "Global.save_epoch_step=1",
            "Global.eval_batch_step=[0,99999]",
            "Global.cal_metric_during_train=False",
            "Global.log_smooth_window=1",
            "Train.loader.num_workers=0",
            "Train.loader.batch_size_per_card=1",
            "Train.loader.drop_last=False",          # KEY FIX: no empty batch crash
            "Eval.loader.num_workers=0",
            "Eval.loader.batch_size_per_card=1",
            "Eval.loader.drop_last=False",            # KEY FIX
        ] + pretrained_args

        print(f"[Train] Starting {version} — {final_count} label entries, 5 epochs")
        proc = await asyncio.create_subprocess_exec(
            *train_cmd, cwd=str(PADDLEOCR_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stdout.decode(errors="replace")[-3000:])

        # ── Step 8: save best_accuracy.pdparams ──────────────────────────────
        saved = sorted(output_dir.glob("*.pdparams"), key=os.path.getmtime)
        if not saved:
            raise FileNotFoundError(f"No .pdparams in {output_dir}")
        best = output_dir / "best_accuracy.pdparams"
        if not best.exists():
            shutil.copy2(saved[-1], best)
            print(f"[Train] best_accuracy saved from {saved[-1].name}")

        # Cleanup optimizer/epoch files
        for f in output_dir.glob("*.pdparams"):
            if f.stem != "best_accuracy": f.unlink(missing_ok=True)
        for f in output_dir.glob("*.pdopt"):
            f.unlink(missing_ok=True)

        # ── Step 9: export inference model ────────────────────────────────────
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
            print(f"[Train] Export warning: {exp_out.decode(errors='replace')[-500:]}")
        else:
            print(f"[Train] Inference exported: {infer_dir}")
            _ocr_cache["version"] = None   # force predictor reload

        # ── Step 10: keep only 2 latest version dirs ──────────────────────────
        all_vdirs = sorted(
            [p for p in MODELS_DIR.glob("v*/") if p.is_dir()],
            key=lambda p: int(p.name.replace("v","")) if p.name.replace("v","").isdigit() else 0
        )
        for old in all_vdirs[:-2]:
            shutil.rmtree(old, ignore_errors=True)
            print(f"[Cleanup] Deleted: {old.name}")

        (MODELS_DIR / f"{version}.pd").touch()
        state["status"]    = "done"
        state["train_end"] = datetime.utcnow().isoformat()
        state["error"]     = None
        print(f"[Train] Complete: {version}")

    except Exception as e:
        state["status"] = "error"
        state["error"]  = str(e)
        print(f"[Train] Error: {e}")

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

        text = find_label_in_history(img_name)
        if not text:
            try:
                text = gemini_extract_text(img_path)
            except Exception as e:
                print(f"[Label] Gemini failed for {img_name}: {e}")
                text = "unknown"
        append_label(label_file, img_name, text)
        state["total_labels_written"] += 1

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


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload one image. Gemini extracts ALL text, OpenCV preprocesses it.
    Training fires automatically after threshold is reached.
    Blocks with 429 while training is running.
    """
    state = load_state()

    if state["status"] == "training":
        raise HTTPException(status_code=429, detail={
            "error":      "Training in progress. Wait until complete.",
            "status":     state["status"],
            "session_id": state["session_id"],
        })

    if (state["status"] == "collecting"
            and len(state["new_images"]) >= NEW_IMAGES_PER_SESSION):
        raise HTTPException(status_code=429, detail={
            "error":  f"Already have {NEW_IMAGES_PER_SESSION} images. Training starting.",
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

    try:
        text = gemini_extract_text(img_path)
    except Exception as e:
        img_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Gemini failed: {e}")

    append_label(Path(state["label_file"]), img_name, text)
    state["new_images"].append(img_name)
    state["total_labels_written"] += 1
    save_state(state)

    collected = len(state["new_images"])
    response  = {
        "session_id":       state["session_id"],
        "model_version":    state["model_version"],
        "image_saved":      img_name,
        "extracted_text":   text,
        "processed": {
            "gray":  proc_result.get("gray"),
            "otsu":  proc_result.get("otsu"),
            "error": proc_result.get("error"),
        },
        "new_images_count": collected,
        "target":           NEW_IMAGES_PER_SESSION,
        "remaining":        max(0, NEW_IMAGES_PER_SESSION - collected),
        "status":           state["status"],
        "message":          f"Image {collected}/{NEW_IMAGES_PER_SESSION} collected.",
    }

    if collected >= NEW_IMAGES_PER_SESSION:
        response["message"] = (
            f"Threshold reached! Sampling {OLD_IMAGES_SAMPLE} old images "
            f"and starting training automatically."
        )
        background_tasks.add_task(prepare_and_train, state)

    return JSONResponse(content=response)


@app.get("/status")
async def get_status():
    """Full system status + all sessions + incremental chain."""
    sessions     = load_all_sessions()
    trained      = sorted(
        [p.parent.name for p in MODELS_DIR.glob("v*/best_accuracy.pdparams")],
        key=lambda x: int(x.replace("v","")) if x.replace("v","").isdigit() else 0,
    )
    label_files  = sorted([p.name for p in LABELS_DIR.glob("label_v*.txt")])
    total_images = len([p for p in IMAGES_DIR.glob("*")
                        if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}])
    total_proc   = len(list(PROCESSED_DIR.glob("*_gray.png")))
    prev_ckpt    = find_previous_checkpoint()
    infer_ready  = find_latest_inference_dir()

    return {
        "total_sessions":    len(sessions),
        "sessions":          sessions,
        "incremental_chain": {
            "trained_versions":  trained,
            "next_version":      f"v{len(trained)+1}",
            "prev_checkpoint":   str(prev_ckpt)+".pdparams" if prev_ckpt else None,
            "inference_ready":   str(infer_ready) if infer_ready else None,
            "chain":             " -> ".join(trained + [f"v{len(trained)+1}(next)"]),
        },
        "storage": {
            "total_images":     total_images,
            "processed_images": total_proc,
            "label_files":      label_files,
            "models_dir":       str(MODELS_DIR),
            "processed_dir":    str(PROCESSED_DIR),
        },
        "thresholds": {
            "new_images_per_session": NEW_IMAGES_PER_SESSION,
            "old_images_sampled":     OLD_IMAGES_SAMPLE,
            "total_before_training":  TOTAL_BEFORE_TRAIN,
            "min_label_entries":      MIN_LABEL_ENTRIES,
        },
    }


@app.get("/status/{session_id}")
async def get_session_status(session_id: str):
    """Detailed status for one session."""
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
        "error":                state["error"],
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


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    tablet_id: str = "default",
):
    """
    OCR using YOUR trained model via paddle.inference predictor.
    No PaddleOCR 3.x API — no model name mismatch errors.
    Returns readings with confidence, bbox, and low-confidence warnings.
    """
    t_start    = time.time()
    request_id = str(uuid.uuid4())

    predictor, char_list, model_version = get_ocr_predictor()
    queue_wait_ms = round((time.time() - t_start) * 1000, 2)

    if predictor is None:
        raise HTTPException(status_code=503, detail=(
            "No trained model available yet. "
            "Upload images to trigger training first, then use /extract."
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
            if not text:
                continue
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