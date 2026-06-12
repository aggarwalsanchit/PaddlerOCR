import os
import re
import sys
import json
import time
import uuid
import yaml
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
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
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
QUEUE_FILE    = DATA_DIR / "upload_queue.json"

# Training trigger thresholds
NEW_IMAGES_PER_SESSION = 1   # collect this many NEW images before firing a session
OLD_IMAGES_SAMPLE      = 1    # random old images added per session
TOTAL_BEFORE_TRAIN     = NEW_IMAGES_PER_SESSION + OLD_IMAGES_SAMPLE

CONFIDENCE_THRESHOLD  = 60.0
TRAIN_TIMEOUT_SECONDS = 1800  # 30 min hard timeout

for _d in [IMAGES_DIR, PROCESSED_DIR, LABELS_DIR,
           MODELS_DIR, DICT_FILE.parent, WORK_DIR, SESSIONS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")

app = FastAPI(title="PaddleOCR Incremental Trainer", version="8.0.0")

_ocr_cache     = {"version": None, "predictor": None, "char_list": None}
_training_proc: Optional[asyncio.subprocess.Process] = None
_training_lock = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# QUEUE  (persisted to disk so it survives server restarts)
# ─────────────────────────────────────────────────────────────────────────────

def queue_load() -> list:
    """Load the pending upload queue from disk. Returns list of image names."""
    if QUEUE_FILE.exists():
        try:
            return json.loads(QUEUE_FILE.read_text())
        except Exception:
            pass
    return []


def queue_save(q: list):
    QUEUE_FILE.write_text(json.dumps(q, indent=2))


def queue_add(image_name: str):
    q = queue_load()
    if image_name not in q:
        q.append(image_name)
        queue_save(q)


def queue_take(n: int) -> list:
    """Remove and return up to n images from the front of the queue."""
    q   = queue_load()
    batch = q[:n]
    queue_save(q[n:])
    return batch


def queue_size() -> int:
    return len(queue_load())


def queue_sessions_remaining() -> int:
    """How many training sessions the current queue will produce."""
    qs = queue_size()
    if qs == 0:
        return 0
    return (qs + NEW_IMAGES_PER_SESSION - 1) // NEW_IMAGES_PER_SESSION


# ─────────────────────────────────────────────────────────────────────────────
# OPENCV PREPROCESSING  (full image — no cropping)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_and_save(img_path: Path) -> dict:
    """
    Grayscale → 4x upscale → bilateral → CLAHE → Otsu.
    Saves full _gray.png and _otsu.png (NO region cropping).
    Training uses the full gray image as input.
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
    pdmodel    = infer_dir / "inference.pdmodel"
    json_m     = infer_dir / "inference.json"
    params     = infer_dir / "inference.pdiparams"
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
    ih  = predictor.get_input_handle(predictor.get_input_names()[0])
    ih.reshape(inp.shape)
    ih.copy_from_cpu(inp)
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
    """Used only for /extract inference — NOT for training."""
    gd = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    g  = cv2.resize(gd, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    g  = cv2.bilateralFilter(g, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g  = clahe.apply(g)
    _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ph, pw   = binary.shape[:2]
    oh, ow   = image.shape[:2]
    sx, sy   = ow / pw, oh / ph
    ia       = pw * ph
    regions  = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 1000 or area > ia * 0.9: continue
        if w < 80 or h < 40: continue
        pad = 20
        x1 = max(0,x-pad);     y1 = max(0,y-pad)
        x2 = min(pw,x+w+pad);  y2 = min(ph,y+h+pad)
        ox1=int(x1*sx); oy1=int(y1*sy); ox2=int(x2*sx); oy2=int(y2*sy)
        crop = image[oy1:oy2, ox1:ox2]
        if crop.size == 0: continue
        regions.append({"bbox":{"x":ox1,"y":oy1,"width":ox2-ox1,"height":oy2-oy1},"crop":crop})
    regions.sort(key=lambda r:r["bbox"]["width"]*r["bbox"]["height"], reverse=True)
    if not regions:
        regions.append({"bbox":{"x":0,"y":0,"width":ow,"height":oh},"crop":image})
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
        "training_phase": None,
        "training_log":   [],
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
    """
    Scan ONLY label files (never deleted) for highest version number.
    This ensures version counter never resets even after model dirs are cleaned.
    """
    existing = set()
    for p in LABELS_DIR.glob("label_v*.txt"):
        part = p.name.replace("label_v","").replace(".txt","")
        try: existing.add(int(part))
        except ValueError: pass
    # Also scan model dirs and session files as secondary sources
    for p in list(MODELS_DIR.glob("v*/")) + list(SESSIONS_DIR.glob("*.json")):
        try:
            if p.suffix == ".json":
                s = json.loads(p.read_text())
                mv = s.get("model_version","")
                if mv and mv.startswith("v"):
                    existing.add(int(mv[1:]))
            else:
                existing.add(int(p.name.replace("v","")))
        except Exception: pass
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
            if p.suffix.lower() in exts and p.name not in exclude]


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI LABELING  (full image text — no cropping)
# ─────────────────────────────────────────────────────────────────────────────

def gemini_extract_text(image_path: Path) -> str:
    """
    Extract ALL text from image as a single line.
    Used for training labels on full gray image.
    """
    import PIL.Image
    img    = PIL.Image.open(image_path)
    prompt = (
        "This image shows a digital display, meter, or instrument panel. "
        "Extract ALL visible text exactly as shown — every number, unit, label, symbol. "
        "Return ONE single line of plain text. "
        "No markdown, no newlines, no explanation. "
        "Example: 238.2V 230.5V 0.0V 57.4kWh"
    )
    response = gemini_model.generate_content([prompt, img])
    raw = response.text.strip()
    raw = raw.replace("```json","").replace("```","").replace("* ","").replace("- ","")
    return " ".join(line.strip() for line in raw.splitlines() if line.strip())


def append_label(label_file: Path, image_name: str, text: str):
    """One entry per image — skips duplicates."""
    abs_img    = str((IMAGES_DIR / image_name).resolve())
    clean_text = " ".join(
        p.strip() for p in text.replace("\r","\n").split("\n") if p.strip()
    ) or "unknown"
    existing = set()
    if label_file.exists():
        for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "\t" in line:
                existing.add(line.split("\t",1)[0].strip())
    if abs_img in existing:
        print(f"[Label] Skip duplicate: {image_name}")
        return
    with open(label_file, "a", encoding="utf-8") as f:
        f.write(f"{abs_img}\t{clean_text}\n")


def find_label_in_history(image_name: str) -> Optional[str]:
    abs_img = str((IMAGES_DIR / image_name).resolve())
    for lf in sorted(LABELS_DIR.glob("label_v*.txt")):
        try:
            for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "\t" not in line: continue
                pp, tp = line.split("\t",1)
                if pp.strip() == abs_img or pp.strip().endswith(image_name):
                    return tp.strip()
        except Exception: continue
    return None


def build_cumulative_label_file() -> Path:
    """
    Merge ALL label entries from every label_vN.txt into one cumulative file.
    This is what gets used for training — the model sees ALL images ever uploaded.
    Returns path to the cumulative file.
    """
    cumulative = LABELS_DIR / "cumulative_all.txt"
    all_entries = {}   # path -> text (deduped, last write wins)

    label_files = sorted(
        LABELS_DIR.glob("label_v*.txt"),
        key=lambda p: int(p.name.replace("label_v","").replace(".txt",""))
        if p.name.replace("label_v","").replace(".txt","").isdigit() else 0
    )
    for lf in label_files:
        try:
            for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "\t" in line:
                    pp, text = line.split("\t",1)
                    pp = pp.strip()
                    if os.path.exists(pp):
                        all_entries[pp] = text.strip()
        except Exception:
            continue

    lines = [f"{p}\t{t}" for p, t in all_entries.items()]
    cumulative.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    print(f"[Train] Cumulative label: {len(lines)} unique images from all sessions")
    return cumulative


# ─────────────────────────────────────────────────────────────────────────────
# LABEL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_fix_label_file(label_file: Path) -> dict:
    """Remove lines with no tab, missing images, or duplicates."""
    if not label_file.exists():
        return {"good": 0, "removed": 0, "removed_lines": []}
    lines     = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    good, bad = [], []
    seen      = set()
    for line in lines:
        line = line.strip()
        if not line: continue
        if "\t" in line:
            pp, _ = line.split("\t",1)
            pp = pp.strip()
            if not os.path.exists(pp):
                bad.append(f"missing_image:{line[:80]}")
            elif pp in seen:
                bad.append(f"duplicate:{pp}")
            else:
                good.append(line); seen.add(pp)
        else:
            bad.append(f"no_tab:{line[:80]}")
    label_file.write_text("\n".join(good) + "\n" if good else "")
    return {"good": len(good), "removed": len(bad), "removed_lines": bad}


def pre_validate_images(label_file: Path) -> dict:
    """Verify images are OpenCV-readable before training — stops RecursionError."""
    if not label_file.exists():
        return {"readable": 0, "skipped": 0, "skipped_paths": []}
    lines   = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    good, skipped = [], []
    for line in lines:
        line = line.strip()
        if not line or "\t" not in line: continue
        pp = line.split("\t",1)[0].strip()
        img = cv2.imread(pp)
        if img is None or img.size == 0:
            skipped.append(pp)
            print(f"[Validate] INVALID removed: {Path(pp).name}")
        else:
            good.append(line)
    if skipped:
        label_file.write_text("\n".join(good) + "\n" if good else "")
    return {"readable": len(good), "skipped": len(skipped), "skipped_paths": skipped}


def prepare_training_label_file(label_file: Path) -> tuple:
    """
    Create temp padded label file (min 8 entries) for PaddleOCR.
    Original stays clean. Returns (tmp_path, padded_count, unique_count).
    """
    unique = [l.strip() for l in
              label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
              if l.strip() and "\t" in l]
    if not unique:
        raise ValueError("No valid entries in label file")
    padded = list(unique)
    while len(padded) < 8:
        padded.extend(unique)
    padded = padded[:max(8, len(unique))]
    tmp = label_file.parent / f"_train_tmp_{label_file.stem}.txt"
    tmp.write_text("\n".join(padded) + "\n", encoding="utf-8")
    print(f"[Train] Temp label: {len(unique)} unique → {len(padded)} padded entries")
    return tmp, len(padded), len(unique)


def patch_simple_dataset():
    """
    Write a PYTHONPATH shim that monkey-patches SimpleDataSet.__getitem__
    to stop at 3 retries instead of recursing 982 times.
    Does NOT modify simple_dataset.py itself.
    """
    guard_path = WORK_DIR / "patch_guard"
    guard_path.mkdir(exist_ok=True)
    shim = guard_path / "ppocr" / "data" / "simple_dataset.py"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(
        "# AUTO-GENERATED recursion guard shim\n"
        "import sys, os\n"
        "sys.path = [p for p in sys.path if str(p) != str(os.path.dirname(os.path.dirname(__file__)))]\n"
        "from ppocr.data.simple_dataset import *\n"
        "from ppocr.data.simple_dataset import SimpleDataSet\n"
        "_orig = SimpleDataSet.__getitem__\n"
        "def _safe(self, idx):\n"
        "    depth = getattr(self, '_pg_depth', 0)\n"
        "    if depth >= 3:\n"
        "        self._pg_depth = 0\n"
        "        return _orig(self, idx % max(1, len(self.data_lines)))\n"
        "    self._pg_depth = depth + 1\n"
        "    try:\n"
        "        return _orig(self, idx)\n"
        "    finally:\n"
        "        self._pg_depth = max(0, getattr(self, '_pg_depth', 1) - 1)\n"
        "try:\n"
        "    SimpleDataSet.__getitem__ = _safe\n"
        "    print('[PatchGuard] SimpleDataSet protected')\n"
        "except Exception as _e:\n"
        "    print(f'[PatchGuard] Warning: {_e}')\n",
        encoding="utf-8"
    )
    print(f"[Patch] Guard shim written to {shim}")
    return str(guard_path)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM YAML CONFIG  (SimpleDataSet — no MultiScaleSampler, no halting)
# ─────────────────────────────────────────────────────────────────────────────

def build_training_yaml(
    label_file: str,
    images_dir: str,
    output_dir: str,
    dict_file:  str,
    pretrained: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 4,
) -> str:
    """
    Write a custom training YAML that uses SimpleDataSet (NOT MultiScaleSampler).
    This is the key fix for training getting stuck/halting forever.
    Returns path to the written YAML file.
    """
    transforms = [
        {"DecodeImage":    {"img_mode": "BGR", "channel_first": False}},
        {"CTCLabelEncode": {}},
        {"RecResizeImg":   {"image_shape": [3, 48, 320]}},
        {"KeepKeys":       {"keep_keys": ["image", "label", "length"]}},
    ]
    cfg = {
        "Global": {
            "use_gpu":                False,
            "epoch_num":              epochs,
            "log_smooth_window":      1,
            "print_batch_step":       10,
            "save_model_dir":         output_dir,
            "save_epoch_step":        max(1, epochs // 2),
            "eval_batch_step":        [0, 50],
            "cal_metric_during_train": True,
            "pretrained_model":       pretrained or "",
            "checkpoints":            None,
            "character_dict_path":    dict_file,
            "max_text_length":        100,
            "infer_mode":             False,
            "use_space_char":         False,
            "distributed":            False,
        },
        "Architecture": {
            "model_type": "rec",
            "algorithm":  "CRNN",
            "Transform":  None,
            "Backbone": {
                "name": "MobileNetV3", "scale": 0.5,
                "model_name": "small", "small_stride": [1,2,2,2],
            },
            "Neck": {"name": "SequenceEncoder", "encoder_type": "rnn", "hidden_size": 48},
            "Head": {"name": "CTCHead", "fc_decay": 4e-4},
        },
        "Loss":        {"name": "CTCLoss"},
        "Optimizer": {
            "name": "Adam", "beta1": 0.9, "beta2": 0.999,
            "lr": {"name": "Cosine", "learning_rate": 0.001, "warmup_epoch": 0},
            "regularizer": {"name": "L2", "factor": 4e-5},
        },
        "PostProcess": {"name": "CTCLabelDecode"},
        "Metric":      {"name": "RecMetric", "main_indicator": "acc"},
        "Train": {
            "dataset": {
                "name": "SimpleDataSet",      # NOT MultiScaleSampler
                "data_dir": images_dir,
                "label_file_list": [label_file],
                "transforms": transforms,
            },
            "loader": {
                "shuffle": True,
                "batch_size_per_card": batch_size,
                "drop_last": False,
                "num_workers": 0,
            },
        },
        "Eval": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": images_dir,
                "label_file_list": [label_file],
                "transforms": transforms,
            },
            "loader": {
                "shuffle": False,
                "drop_last": False,
                "batch_size_per_card": batch_size,
                "num_workers": 0,
            },
        },
    }
    cfg_path = WORK_DIR / "train_config.yml"
    cfg_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    return str(cfg_path)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_training_metrics(log_file: Path) -> dict:
    if not log_file.exists():
        return {"accuracy": None, "loss": None, "best_acc": None}
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    loss = accuracy = best_acc = None
    loss_m = re.findall(r"CTCLoss:\s*([\d.]+)", text)
    if loss_m:
        try: loss = round(float(loss_m[-1]), 4)
        except ValueError: pass
    acc_m = re.findall(r"\bacc:\s*([\d.]+)", text)
    if acc_m:
        try:
            vals = [float(v) for v in acc_m]
            last = vals[-1]
            accuracy = round(last*100,2) if last<=1.0 else round(last,2)
            mx = max(vals)
            best_acc = round(mx*100,2) if mx<=1.0 else round(mx,2)
        except ValueError: pass
    best_m = re.findall(r"best metric,\s*acc:\s*([\d.]+)", text)
    if best_m:
        try:
            v = float(best_m[-1])
            best_acc = round(v*100,2) if v<=1.0 else round(v,2)
        except ValueError: pass
    return {"accuracy": accuracy, "loss": loss, "best_acc": best_acc}


def get_all_model_scores() -> list:
    scores = []
    dirs = sorted(
        [p for p in MODELS_DIR.glob("v*/") if p.is_dir()],
        key=lambda p: int(p.name.replace("v","")) if p.name.replace("v","").isdigit() else 0
    )
    for vdir in dirs:
        m = parse_training_metrics(vdir / "train.log")
        scores.append({
            "version":      vdir.name,
            "has_model":    (vdir / "best_accuracy.pdparams").exists(),
            "has_inference":(vdir / "inference" / "inference.pdiparams").exists(),
            "accuracy_pct": m["accuracy"],
            "best_acc_pct": m["best_acc"],
            "final_loss":   m["loss"],
        })
    return scores


def count_total_unique_trained_images() -> int:
    all_paths = set()
    for s in load_all_sessions():
        if s.get("status") != "done": continue
        lf = s.get("label_file")
        if not lf or not Path(lf).exists(): continue
        for line in Path(lf).read_text(encoding="utf-8", errors="ignore").splitlines():
            if "\t" in line:
                pp = line.split("\t",1)[0].strip()
                if os.path.exists(pp):
                    all_paths.add(pp)
    return len(all_paths)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING  (incremental, cumulative, custom YAML, auto-fires from queue)
# ─────────────────────────────────────────────────────────────────────────────

async def run_training(state: dict):
    global _training_proc
    version    = state["model_version"]
    label_file = Path(state["label_file"])
    output_dir = MODELS_DIR / version

    state["status"]         = "training"
    state["train_start"]    = datetime.utcnow().isoformat()
    state["error"]          = None
    state["training_log"]   = []
    state["training_phase"] = "initializing"
    save_state(state)

    def _log(line: str, phase: str = None):
        print(line)
        state["training_log"].append(line)
        if len(state["training_log"]) > 50:
            state["training_log"] = state["training_log"][-50:]
        if phase:
            state["training_phase"] = phase

    tmp_label = None
    cfg_file  = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: build cumulative label (ALL images ever uploaded)
        _log("[Train] Step 1/8 — Building cumulative label file...", phase="building_cumulative")
        save_state(state)
        cumulative_file = build_cumulative_label_file()

        # Step 2: validate cumulative file
        _log("[Train] Step 2/8 — Validating labels...", phase="validating_labels")
        fix = validate_and_fix_label_file(cumulative_file)
        _log(f"[Train] Labels: {fix['good']} good, {fix['removed']} removed")
        save_state(state)
        if fix["good"] == 0:
            raise ValueError("0 valid label entries after cleanup.")

        # Step 3: pre-validate images
        _log("[Train] Step 3/8 — Validating images...", phase="validating_images")
        img_check = pre_validate_images(cumulative_file)
        _log(f"[Train] Images: {img_check['readable']} readable, {img_check['skipped']} invalid removed")
        save_state(state)
        if img_check["readable"] == 0:
            raise ValueError("All images are unreadable/corrupt.")

        # Step 4: pad to min 8
        _log("[Train] Step 4/8 — Preparing temp label file...", phase="preparing_labels")
        tmp_label, train_count, unique_count = prepare_training_label_file(cumulative_file)
        state["total_labels_written"] = unique_count
        _log(f"[Train] Training on {unique_count} unique images ({train_count} padded entries)")
        save_state(state)

        # Step 5: clone PaddleOCR if needed
        if not PADDLEOCR_DIR.exists():
            _log("[Train] Step 5/8 — Cloning PaddleOCR...", phase="cloning_paddleocr")
            save_state(state)
            r = await asyncio.create_subprocess_exec(
                "git","clone","--depth","1",
                "https://github.com/PaddlePaddle/PaddleOCR.git",
                str(PADDLEOCR_DIR),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await r.communicate()
            if r.returncode != 0:
                raise RuntimeError(f"Clone failed: {out.decode()}")
            _log("[Train] PaddleOCR cloned OK")
        else:
            _log("[Train] Step 5/8 — PaddleOCR already present")

        # Step 6: setup recursion guard shim
        _log("[Train] Step 6/8 — Setting up recursion guard...", phase="patching")
        guard_dir = patch_simple_dataset()
        save_state(state)

        # Step 7: build custom YAML config (SimpleDataSet — no halting)
        _log("[Train] Step 7/8 — Writing custom YAML config (SimpleDataSet)...", phase="configuring")
        prev_ckpt = find_previous_checkpoint()
        _log(f"[Train] Prev checkpoint: {prev_ckpt or 'none (fresh start)'}")

        label_abs  = str(tmp_label.resolve())
        images_abs = str(IMAGES_DIR.resolve())
        output_abs = str(output_dir.resolve())
        dict_abs   = str(DICT_FILE.resolve())

        # Auto-scale batch size based on dataset size
        batch = min(4, max(1, unique_count // 4))
        cfg_file = build_training_yaml(
            label_file = label_abs,
            images_dir = images_abs,
            output_dir = output_abs,
            dict_file  = dict_abs,
            pretrained = str(prev_ckpt) if prev_ckpt else None,
            epochs     = 10,
            batch_size = batch,
        )
        _log(f"[Train] Config: SimpleDataSet, batch={batch}, epochs=10")

        # Step 8: run training with timeout
        _log(f"[Train] Step 8/8 — Launching training ({unique_count} images, timeout={TRAIN_TIMEOUT_SECONDS}s)...",
             phase="training")
        save_state(state)

        train_env = os.environ.copy()
        ep = train_env.get("PYTHONPATH","")
        train_env["PYTHONPATH"] = (guard_dir + os.pathsep + ep) if ep else guard_dir

        train_script = PADDLEOCR_DIR / "tools" / "train.py"
        _training_proc = await asyncio.create_subprocess_exec(
            sys.executable, str(train_script), "-c", cfg_file,
            cwd=str(PADDLEOCR_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=train_env,
        )

        train_start_ts = time.time()
        rc = None
        while True:
            elapsed   = time.time() - train_start_ts
            remaining = TRAIN_TIMEOUT_SECONDS - elapsed
            if remaining <= 0:
                _log(f"[Train] TIMEOUT after {TRAIN_TIMEOUT_SECONDS}s — killing")
                try:
                    _training_proc.terminate()
                    await asyncio.sleep(2)
                    _training_proc.kill()
                except Exception: pass
                raise RuntimeError(f"Training timed out after {TRAIN_TIMEOUT_SECONDS}s.")
            try:
                line_bytes = await asyncio.wait_for(
                    _training_proc.stdout.readline(),
                    timeout=min(30.0, remaining)
                )
            except asyncio.TimeoutError:
                elapsed_i = int(time.time() - train_start_ts)
                _log(f"[Train] ... still running ({elapsed_i}s, no output in 30s)")
                save_state(state)
                await asyncio.sleep(0)
                continue
            if not line_bytes:
                break
            line = line_bytes.decode(errors="replace").rstrip()
            _log(line)
            save_state(state)
            await asyncio.sleep(0)

        rc = await _training_proc.wait()
        _training_proc = None

        if rc is not None and rc != 0 and rc != -15:
            raise RuntimeError(f"train.py exited code {rc}. Check training_log.")
        if rc == -15:
            raise RuntimeError("Stopped manually via POST /stop.")

        # Save best_accuracy.pdparams
        _log("[Train] Saving best checkpoint...", phase="saving_checkpoint")
        saved = sorted(output_dir.glob("*.pdparams"), key=os.path.getmtime)
        if not saved:
            raise FileNotFoundError(f"No .pdparams in {output_dir}")
        best = output_dir / "best_accuracy.pdparams"
        if not best.exists():
            shutil.copy2(saved[-1], best)
            _log(f"[Train] best_accuracy saved from {saved[-1].name}")

        for f in output_dir.glob("*.pdparams"):
            if f.stem != "best_accuracy": f.unlink(missing_ok=True)
        for f in output_dir.glob("*.pdopt"):
            f.unlink(missing_ok=True)

        # Parse metrics
        m = parse_training_metrics(output_dir / "train.log")
        state["accuracy"] = m.get("best_acc") or m.get("accuracy")
        state["loss"]     = m.get("loss")
        _log(f"[Train] Accuracy: {state['accuracy']}%  Loss: {state['loss']}")

        # Export inference model
        _log("[Train] Exporting inference model...", phase="exporting")
        save_state(state)
        infer_dir = output_dir / "inference"
        infer_dir.mkdir(exist_ok=True)

        exp_cfg = yaml.safe_load(Path(cfg_file).read_text())
        exp_cfg["Global"]["pretrained_model"]  = str(output_dir / "best_accuracy")
        exp_cfg["Global"]["save_inference_dir"] = str(infer_dir)
        exp_cfg_path = WORK_DIR / "export_config.yml"
        exp_cfg_path.write_text(yaml.dump(exp_cfg, default_flow_style=False))

        exp_script = PADDLEOCR_DIR / "tools" / "export_model.py"
        exp = await asyncio.create_subprocess_exec(
            sys.executable, str(exp_script), "-c", str(exp_cfg_path),
            cwd=str(PADDLEOCR_DIR),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env=train_env,
        )
        try:
            exp_out, _ = await asyncio.wait_for(exp.communicate(), timeout=120)
            if exp.returncode != 0:
                _log(f"[Train] Export warning: {exp_out.decode(errors='replace')[-200:]}")
            else:
                _ocr_cache["version"] = None
                _log(f"[Train] Inference exported: {infer_dir}")
        except asyncio.TimeoutError:
            exp.kill()
            _log("[Train] Export timed out (non-fatal)")

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
        if tmp_label and tmp_label.exists():
            tmp_label.unlink(missing_ok=True)
        _training_proc = None

    save_state(state)


async def fire_next_session_from_queue(background_tasks: BackgroundTasks):
    """
    Take the next batch from the queue and start a training session.
    Called automatically after each session completes.
    """
    if queue_size() == 0:
        print("[Queue] Empty — no more sessions to fire")
        return

    batch = queue_take(NEW_IMAGES_PER_SESSION)
    if not batch:
        return

    # Sample old images
    old_available = get_old_images(exclude=batch)
    sampled       = random.sample(old_available, min(OLD_IMAGES_SAMPLE, len(old_available)))

    version    = next_version()
    session_id = f"session_{version}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    label_path = (LABELS_DIR / f"label_{version}.txt").resolve()
    label_path.touch()

    state = _default_state()
    state.update({
        "session_id":    session_id,
        "status":        "collecting",
        "model_version": version,
        "label_file":    str(label_path),
        "new_images":    batch,
        "sampled_old":   sampled,
    })

    # Label new images
    for img_name in batch:
        img_path = IMAGES_DIR / img_name
        text     = find_label_in_history(img_name)
        if not text:
            try:
                # Use gray image for labeling
                gray_path = PROCESSED_DIR / f"{img_path.stem}_gray.png"
                label_src = gray_path if gray_path.exists() else img_path
                text = gemini_extract_text(label_src)
            except Exception as e:
                print(f"[Label] Gemini failed for {img_name}: {e}")
                text = "unknown"
        append_label(label_path, img_name, text)

    # Label sampled old images
    for img_name in sampled:
        img_path  = IMAGES_DIR / img_name
        gray_path = PROCESSED_DIR / f"{img_path.stem}_gray.png"
        otsu_path = PROCESSED_DIR / f"{img_path.stem}_otsu.png"
        if not gray_path.exists() or not otsu_path.exists():
            preprocess_and_save(img_path)
        text = find_label_in_history(img_name)
        if not text:
            try:
                label_src = gray_path if gray_path.exists() else img_path
                text = gemini_extract_text(label_src)
            except Exception as e:
                text = "unknown"
        append_label(label_path, img_name, text)

    state["total_labels_written"] = len(batch) + len(sampled)
    save_state(state)

    print(f"[Queue] Firing session {session_id} — {len(batch)} new + {len(sampled)} old, {queue_size()} remaining in queue")
    background_tasks.add_task(_run_and_chain, state, background_tasks)


async def _run_and_chain(state: dict, background_tasks: BackgroundTasks):
    """Run training then automatically fire next session if queue has more."""
    async with _training_lock:
        await run_training(state)

    # After training completes, fire next session if queue still has images
    if queue_size() >= 1:
        print(f"[Queue] {queue_size()} images remaining — firing next session")
        await fire_next_session_from_queue(background_tasks)


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

if PADDLEOCR_DIR.exists():
    patch_simple_dataset()
    print("[Startup] Recursion guard shim ready.")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload one image. Always accepted — goes into queue.
    If queue reaches NEW_IMAGES_PER_SESSION threshold, training fires automatically.
    Never blocks — even if training is running, uploads are queued.
    """
    # Save image
    suffix   = Path(file.filename or "image.jpg").suffix.lower() or ".jpg"
    img_name = f"img_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}{suffix}"
    img_path = IMAGES_DIR / img_name
    img_path.write_bytes(await file.read())

    # Preprocess (gray + otsu full image — no cropping)
    proc_result = preprocess_and_save(img_path)

    # Gemini extract text (uses gray image)
    try:
        gray_path = PROCESSED_DIR / f"{img_path.stem}_gray.png"
        label_src = gray_path if gray_path.exists() else img_path
        text = gemini_extract_text(label_src)
    except Exception as e:
        img_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Gemini failed: {e}")

    # Add to queue
    queue_add(img_name)
    qs = queue_size()

    current_state = load_state()
    is_training   = current_state.get("status") == "training"

    response = {
        "session_id":       current_state.get("session_id"),
        "model_version":    current_state.get("model_version"),
        "image_saved":      img_name,
        "extracted_text":   text,
        "region_crops":     [],                          # kept for API compatibility
        "labels_written":   1,
        "processed": {
            "gray":  proc_result.get("gray"),
            "otsu":  proc_result.get("otsu"),
            "error": proc_result.get("error"),
        },
        "new_images_count": qs,
        "target":           NEW_IMAGES_PER_SESSION,
        "remaining":        max(0, NEW_IMAGES_PER_SESSION - qs),
        "status":           "queued_training_running" if is_training else "collecting",
        "queue": {
            "pending_images":      qs,
            "sessions_remaining":  queue_sessions_remaining(),
            "training_running":    is_training,
            "will_auto_train_at":  NEW_IMAGES_PER_SESSION,
        },
        "message": (
            f"Image queued. Queue: {qs} images. Training running — will fire after current session."
            if is_training else
            f"Image {qs}/{NEW_IMAGES_PER_SESSION} in queue."
        ),
    }

    # Fire training if threshold reached and not currently training
    if qs >= NEW_IMAGES_PER_SESSION and not is_training:
        response["message"] = (
            f"Queue threshold reached ({qs} images)! "
            f"Starting training session automatically."
        )
        background_tasks.add_task(fire_next_session_from_queue, background_tasks)

    return JSONResponse(content=response)


@app.get("/status")
async def get_status():
    """
    Full status — all existing keys preserved + new queue and cumulative fields added.
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
    current        = load_state()
    qs             = queue_size()
    is_training    = current.get("status") == "training"

    elapsed_sec = None
    if is_training and current.get("train_start"):
        try:
            started     = datetime.fromisoformat(current["train_start"])
            elapsed_sec = int((datetime.utcnow() - started).total_seconds())
        except Exception: pass

    return {
        # ── All original keys preserved ────────────────────────────────────────
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

        # ── NEW fields added (do not change existing keys above) ───────────────
        "queue": {
            "pending_images":     qs,
            "sessions_remaining": queue_sessions_remaining(),
            "training_running":   is_training,
            "will_auto_train_at": NEW_IMAGES_PER_SESSION,
            "note": (
                f"{qs} images queued. {queue_sessions_remaining()} more sessions will fire automatically."
                if qs > 0 else "Queue is empty."
            ),
        },
        "training_now": {
            "is_training":       is_training,
            "elapsed_seconds":   elapsed_sec,
            "timeout_seconds":   TRAIN_TIMEOUT_SECONDS,
            "training_phase":    current.get("training_phase"),
            "training_on_images": current.get("total_labels_written", 0),
            "live_log_last_5":   current.get("training_log", [])[-5:],
            "config_type":       "SimpleDataSet (no MultiScaleSampler — no halting)",
            "uses_full_gray_image": True,
            "no_region_cropping":   True,
        },
        "cumulative_training": {
            "total_unique_images_trained": total_trained,
            "cumulative_label_exists":     (LABELS_DIR / "cumulative_all.txt").exists(),
            "note": "Each session trains on ALL images ever uploaded, not just the latest batch.",
        },
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
    """Kill stuck training, reset to idle."""
    global _training_proc

    stopped_pid = None
    if _training_proc is not None:
        try:
            _training_proc.terminate()
            stopped_pid = _training_proc.pid
            print(f"[Stop] SIGTERM → PID {stopped_pid}")
            await asyncio.sleep(2)
            try: _training_proc.kill()
            except Exception: pass
        except Exception as e:
            print(f"[Stop] Error: {e}")
        _training_proc = None

    try:
        subprocess.run(
            ["pkill", "-u", os.environ.get("USER",""), "-f", "train.py"],
            capture_output=True
        )
    except Exception: pass

    reset_count = 0
    for sf in SESSIONS_DIR.glob("*.json"):
        try:
            s = json.loads(sf.read_text())
            if s.get("status") in ("training","collecting"):
                s["status"]    = "error"
                s["error"]     = "Stopped manually via POST /stop"
                s["train_end"] = datetime.utcnow().isoformat()
                sf.write_text(json.dumps(s, indent=2))
                reset_count += 1
        except Exception: continue

    for tmp in LABELS_DIR.glob("_train_tmp_*.txt"):
        tmp.unlink(missing_ok=True)

    if PADDLEOCR_DIR.exists():
        ds_path = PADDLEOCR_DIR / "ppocr" / "data" / "simple_dataset.py"
        if ds_path.exists():
            ds_src = ds_path.read_text(encoding="utf-8")
            if "PATCHED_V7_RECURSION_GUARD" in ds_src or "AUTO-GENERATED by main.py" in ds_src:
                import subprocess as _sp
                _sp.run(["git","checkout","ppocr/data/simple_dataset.py"],
                        cwd=str(PADDLEOCR_DIR), capture_output=True)
        patch_simple_dataset()

    return {
        "message":        "Training stopped. System reset to idle.",
        "stopped_pid":    stopped_pid,
        "sessions_reset": reset_count,
        "queue_size":     queue_size(),
        "next_action":    "POST /upload to add images, or POST /flush-queue to train with current queue",
    }


@app.post("/flush-queue")
async def flush_queue(background_tasks: BackgroundTasks):
    """
    Force start training with whatever images are currently in queue,
    even if below the NEW_IMAGES_PER_SESSION threshold.
    Useful when you want to train immediately without waiting for 20 images.
    """
    qs = queue_size()
    if qs == 0:
        return {"message": "Queue is empty. Upload images first.", "queue_size": 0}

    current = load_state()
    if current.get("status") == "training":
        return {
            "message": f"Training already running. {qs} images will be trained in next session.",
            "queue_size": qs
        }

    background_tasks.add_task(fire_next_session_from_queue, background_tasks)
    return {
        "message":   f"Forcing training with {qs} queued images.",
        "queue_size": qs,
        "note":      "Training will start in background. Poll GET /status for progress.",
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
            "No trained model yet. Upload images to trigger training first."
        ))

    suffix   = Path(file.filename or "img.jpg").suffix.lower() or ".jpg"
    tmp_path = DATA_DIR / f"_extract_{request_id}{suffix}"
    try:
        tmp_path.write_bytes(await file.read())
        image_bgr = cv2.imread(str(tmp_path))
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Cannot read image.")

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
                    "warning":    f"Confidence {confidence:.1f}% below {CONFIDENCE_THRESHOLD:.1f}%",
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

    except HTTPException: raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
