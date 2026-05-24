"""
PaddleOCR Incremental Training System with Gemini Auto-Labeling
4 Endpoints:
  POST /upload        - Upload image one by one (Gemini labels it automatically)
  GET  /status        - Overall system status + history
  GET  /status/{sid}  - Single session detail
  POST /extract       - Test Gemini extraction without affecting training

Flow:
  1. User uploads images one by one
  2. Each image -> Gemini extracts text (single line) -> appended to label_vN.txt
  3. After NEW_IMAGES_PER_SESSION images -> sample OLD_IMAGES_SAMPLE old images
  4. Total = NEW + OLD -> training starts automatically
  5. Training uses previous best_accuracy.pdparams (v1->v2->v3 incremental chain)
  6. Processed images saved as _gray.png + _otsu.png in data/processed/
"""

import os
import sys
import json
import random
import shutil
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCaweP4mlYoLC1BzZF88MBwVp6gdsT_Ahk")

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

NEW_IMAGES_PER_SESSION = 1
OLD_IMAGES_SAMPLE      = 1
TOTAL_BEFORE_TRAIN     = NEW_IMAGES_PER_SESSION + OLD_IMAGES_SAMPLE

for _d in [IMAGES_DIR, PROCESSED_DIR, LABELS_DIR,
           MODELS_DIR, DICT_FILE.parent, WORK_DIR, SESSIONS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")

app = FastAPI(title="PaddleOCR Incremental Trainer", version="3.1.0")
OCR_MODEL = None
OCR_MODEL_VERSION = None

# ─────────────────────────────────────────────────────────────────────────────
# OPENCV PREPROCESSING
# Saves _gray.png (CLAHE enhanced) and _otsu.png (binary) for every image
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_and_save(img_path: Path) -> dict:
    """
    Run exact OpenCV pipeline from original image_processing.py.
    Saves both variants to data/processed/.
    Returns {"gray": path, "otsu": path} or {"error": msg}
    """
    try:
        image = cv2.imread(str(img_path))
        if image is None:
            return {"error": f"Cannot load: {img_path.name}"}

        stem = img_path.stem

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 4x upscale
        gray = cv2.resize(gray, None, fx=4, fy=4,
                          interpolation=cv2.INTER_CUBIC)

        # Bilateral filter (edge-preserving noise removal)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # CLAHE (adaptive contrast enhancement)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # Otsu binary threshold
        _, otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        gray_path = PROCESSED_DIR / f"{stem}_gray.png"
        otsu_path = PROCESSED_DIR / f"{stem}_otsu.png"
        cv2.imwrite(str(gray_path), gray)
        cv2.imwrite(str(otsu_path), otsu)

        return {"gray": str(gray_path), "otsu": str(otsu_path)}

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def _default_state() -> dict:
    return {
        "session_id":           None,
        "status":               "idle",
        "new_images":           [],
        "sampled_old":          [],
        "label_file":           None,
        "model_version":        None,
        "train_start":          None,
        "train_end":            None,
        "error":                None,
        "total_labels_written": 0,
    }

def get_session_file(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def load_state(session_id: Optional[str] = None) -> dict:
    """
    Load latest session or specific session.
    """

    if session_id:
        sf = get_session_file(session_id)

        if sf.exists():
            try:
                return json.loads(sf.read_text())
            except Exception:
                return _default_state()

        return _default_state()

    # latest session
    session_files = sorted(
        SESSIONS_DIR.glob("*.json"),
        key=os.path.getmtime,
        reverse=True
    )

    if not session_files:
        return _default_state()

    try:
        return json.loads(session_files[0].read_text())
    except Exception:
        return _default_state()


def save_state(state: dict):
    """
    Save each session separately.
    """

    session_id = state.get("session_id")

    if not session_id:
        return

    sf = get_session_file(session_id)

    sf.write_text(
        json.dumps(state, indent=2, default=str)
    )


def load_all_sessions():
    """
    Load all sessions sorted newest first.
    """

    sessions = []

    session_files = sorted(
        SESSIONS_DIR.glob("*.json"),
        key=os.path.getmtime,
        reverse=True
    )

    for sf in session_files:
        try:
            data = json.loads(sf.read_text())
            sessions.append(data)
        except Exception:
            continue

    return sessions
# def load_state() -> dict:
#     if SESSION_FILE.exists():
#         try:
#             return json.loads(SESSION_FILE.read_text())
#         except Exception:
#             pass
#     return _default_state()


# def save_state(state: dict):
#     SESSION_FILE.write_text(json.dumps(state, indent=2, default=str))


def next_version() -> str:
    """Find next vN by scanning model dirs and label files."""
    existing = set()
    for p in list(MODELS_DIR.glob("v*/")) + list(LABELS_DIR.glob("label_v*.txt")):
        name = p.name
        part = name.replace("label_", "").replace(".txt", "").replace("v", "")
        try:
            existing.add(int(part))
        except ValueError:
            pass
    return f"v{max(existing, default=0) + 1}"


def find_previous_checkpoint() -> Optional[Path]:
    """
    Find best_accuracy.pdparams from the most recent successful training.
    Returns path WITHOUT .pdparams extension (PaddleOCR adds it automatically).
    Enables v1 -> v2 -> v3 incremental chain.
    """
    version_dirs = sorted(
        [p for p in MODELS_DIR.glob("v*/") if p.is_dir()],
        key=lambda p: int(p.name.replace("v", ""))
        if p.name.replace("v", "").isdigit() else 0
    )
    for vdir in reversed(version_dirs):
        best = vdir / "best_accuracy.pdparams"
        if best.exists():
            return vdir / "best_accuracy"
    return None


def get_old_images(exclude: list) -> list:
    """All images in IMAGES_DIR not in current session."""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    all_imgs = [p.name for p in IMAGES_DIR.glob("*")
                if p.suffix.lower() in exts]
    return [img for img in all_imgs if img not in exclude]


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI LABELING
# ─────────────────────────────────────────────────────────────────────────────

def gemini_extract_text(image_path: Path) -> str:
    """
    Use Gemini Vision to extract text from a digital display.
    Forces single-line output — multi-line breaks PaddleOCR label format.
    """
    import PIL.Image
    img = PIL.Image.open(image_path)
    prompt = (
        "You are an OCR labeling assistant for training a PaddleOCR model on digital displays. "
        "Look at this image and extract the main value shown — numbers, units, symbols "
        "(e.g. 23.5°C, 1234, 56.7V, 100%). "
        "IMPORTANT: Return ONLY a single line of text. "
        "No newlines, no bullet points, no explanations, no markdown. "
        "Just the raw value as one single line."
    )
    response = gemini_model.generate_content([prompt, img])
    # Safety: take only first non-empty line even if Gemini ignores instructions
    for line in response.text.strip().split("\n"):
        line = line.strip()
        if line:
            return line
    return response.text.strip()


def append_label(label_file: Path, image_name: str, text: str):
    """
    Append PaddleOCR label line.
    Format: /absolute/path/to/image.jpg\ttext  (MUST be single line)

    Fixes multi-line Gemini responses by joining all lines with a space.
    This prevents orphan lines like:
        Alarm
        Furnace 1
        24.00
    from appearing in the label file without image paths.
    """
    abs_img = str((IMAGES_DIR / image_name).resolve())

    # Join all lines into one — PaddleOCR cannot handle newlines in label text
    clean_text = " ".join(
        part.strip()
        for part in text.replace("\r", "\n").split("\n")
        if part.strip()
    )

    if not clean_text:
        clean_text = "unknown"

    with open(label_file, "a", encoding="utf-8") as f:
        f.write(f"{abs_img}\t{clean_text}\n")


def find_label_in_history(image_name: str) -> Optional[str]:
    """Search all previous label files for this image's label."""
    abs_img = str((IMAGES_DIR / image_name).resolve())
    for lf in sorted(LABELS_DIR.glob("label_v*.txt")):
        try:
            for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "\t" not in line:
                    continue
                path_part, text_part = line.split("\t", 1)
                if path_part.strip() == abs_img or path_part.strip().endswith(image_name):
                    return text_part.strip()
        except Exception:
            continue
    return None


def validate_and_fix_label_file(label_file: Path) -> dict:
    """
    Read label file and remove any lines without tab separator.
    These orphan lines (from multi-line Gemini responses) cause PaddleOCR
    IndexError: list index out of range.
    Returns {"good": int, "removed": int, "removed_lines": list}
    """
    if not label_file.exists():
        return {"good": 0, "removed": 0, "removed_lines": []}

    lines        = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    good_lines   = []
    removed      = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            path_part, text_part = line.split("\t", 1)
            # Also validate image file exists
            if os.path.exists(path_part.strip()):
                good_lines.append(line)
            else:
                removed.append(f"missing_image: {line[:80]}")
        else:
            removed.append(f"no_tab: {line[:80]}")

    label_file.write_text("\n".join(good_lines) + "\n" if good_lines else "")
    return {
        "good":          len(good_lines),
        "removed":       len(removed),
        "removed_lines": removed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# Incremental chain: each session loads previous best_accuracy.pdparams
# ─────────────────────────────────────────────────────────────────────────────

async def run_training(state: dict):
    """Run PaddleOCR training subprocess with incremental checkpoint loading."""
    version    = state["model_version"]
    label_file = Path(state["label_file"])
    output_dir = MODELS_DIR / version

    state["status"]      = "training"
    state["train_start"] = datetime.utcnow().isoformat()
    state["error"]       = None
    save_state(state)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Validate label file before training ───────────────────────────────
        fix_result = validate_and_fix_label_file(label_file)
        print(f"[Training] Label file: {fix_result['good']} good, "
              f"{fix_result['removed']} removed")
        if fix_result["removed"]:
            for r in fix_result["removed_lines"]:
                print(f"  Removed: {r}")

        if fix_result["good"] == 0:
            raise ValueError(
                "Label file has 0 valid entries after cleanup. "
                "Check that images exist and labels are tab-separated."
            )

        # ── Clone PaddleOCR repo if not present ───────────────────────────────
        if not PADDLEOCR_DIR.exists():
            print("[Training] Cloning PaddleOCR repo...")
            clone = await asyncio.create_subprocess_exec(
                "git", "clone",
                "https://github.com/PaddlePaddle/PaddleOCR.git",
                str(PADDLEOCR_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await clone.communicate()
            if clone.returncode != 0:
                raise RuntimeError(f"Git clone failed:\n{out.decode()}")
            print("[Training] PaddleOCR cloned OK")

        # ── Find previous checkpoint for incremental training ─────────────────
        prev_ckpt = find_previous_checkpoint()
        if prev_ckpt:
            pretrained_args = [f"Global.pretrained_model={str(prev_ckpt)}"]
            print(f"[Training] Resuming from: {prev_ckpt}.pdparams")
        else:
            pretrained_args = []
            print("[Training] No previous checkpoint — training from scratch")

        # ── All paths as absolute strings ──────────────────────────────────────
        label_abs  = str(label_file.resolve())
        images_abs = str(IMAGES_DIR.resolve())
        output_abs = str(output_dir.resolve())
        dict_abs   = str(DICT_FILE.resolve())

        train_cmd = [
            sys.executable,
            "tools/train.py",
            "-c", "configs/rec/PP-OCRv3/en_PP-OCRv3_mobile_rec.yml",
            "-o",
            f"Train.dataset.data_dir={images_abs}",
            f"Train.dataset.label_file_list=['{label_abs}']",
            f"Eval.dataset.data_dir={images_abs}",
            f"Eval.dataset.label_file_list=['{label_abs}']",
            f"Global.save_model_dir={output_abs}",
            f"Global.character_dict_path={dict_abs}",
            "Global.use_gpu=False",
            "Global.epoch_num=5",
            "Global.save_epoch_step=1",          # save every epoch
            "Global.eval_batch_step=[0,99999]",  # never eval — saves time
            "Global.cal_metric_during_train=False",
            "Global.log_smooth_window=1",
            "Train.loader.num_workers=0",
            "Train.loader.batch_size_per_card=2",
            "Eval.loader.num_workers=0",
            "Eval.loader.batch_size_per_card=2",
        ] + pretrained_args

        print(f"[Training] Starting: {version}, {fix_result['good']} images, 5 epochs")

        proc = await asyncio.create_subprocess_exec(
            *train_cmd,
            cwd=str(PADDLEOCR_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output_text = stdout.decode(errors="replace")

        if proc.returncode != 0:
            raise RuntimeError(output_text[-3000:])

        # ── Find and rename saved model to best_accuracy.pdparams ─────────────
        # PaddleOCR saves as 'latest.pdparams' or 'iter_epoch_N.pdparams'
        # when eval is disabled (no best_accuracy is written)
        saved_params = sorted(
            list(output_dir.glob("*.pdparams")),
            key=os.path.getmtime
        )

        if not saved_params:
            raise FileNotFoundError(
                f"Training finished but no .pdparams in {output_dir}. "
                f"Check {output_dir}/train.log"
            )

        best = output_dir / "best_accuracy.pdparams"
        if not best.exists():
            # Rename the latest saved epoch to best_accuracy
            shutil.copy2(saved_params[-1], best)
            print(f"[Training] Saved best_accuracy from: {saved_params[-1].name}")
        else:
            print(f"[Training] best_accuracy.pdparams already exists")

        # ── Cleanup: delete optimizer states + old epoch files ────────────────
        for f in output_dir.glob("*.pdparams"):
            if f.stem != "best_accuracy":
                f.unlink(missing_ok=True)
        for f in output_dir.glob("*.pdopt"):
            f.unlink(missing_ok=True)

        # ── Keep only latest 2 version dirs to save disk space ────────────────
        all_vdirs = sorted(
            [p for p in MODELS_DIR.glob("v*/") if p.is_dir()],
            key=lambda p: int(p.name.replace("v", ""))
            if p.name.replace("v", "").isdigit() else 0
        )
        for old in all_vdirs[:-2]:
            shutil.rmtree(old, ignore_errors=True)
            print(f"[Cleanup] Deleted old model dir: {old.name}")

        # ── Version marker ────────────────────────────────────────────────────
        (MODELS_DIR / f"{version}.pd").touch()

        state["status"]    = "done"
        state["train_end"] = datetime.utcnow().isoformat()
        state["error"]     = None
        
        # Export inference model
        export_cmd = [
            sys.executable,
            "tools/export_model.py",
            "-c", "configs/rec/PP-OCRv3/en_PP-OCRv3_mobile_rec.yml",
            "-o",
            f"Global.pretrained_model={str(output_dir / 'best_accuracy')}",
            f"Global.save_inference_dir={str(output_dir / 'inference')}",
        ]
        
        export_proc = await asyncio.create_subprocess_exec(
            *export_cmd,
            cwd=str(PADDLEOCR_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        
        export_out, _ = await export_proc.communicate()
        
        if export_proc.returncode != 0:
            raise RuntimeError(
                f"Model export failed:\n{export_out.decode(errors='replace')}"
            )
        
        print("[Training] Inference model exported")
        print(f"[Training] Complete: {output_dir}/best_accuracy.pdparams")

    except Exception as e:
        state["status"] = "error"
        state["error"]  = str(e)
        print(f"[Training] Error: {e}")

    save_state(state)


async def prepare_and_train(state: dict):
    """
    1. Sample OLD_IMAGES_SAMPLE old images
    2. Look up labels from history or re-extract with Gemini
    3. Append to label file
    4. Validate label file (remove bad lines)
    5. Start training
    """
    old_available = get_old_images(exclude=state["new_images"])
    sampled       = random.sample(old_available,
                                  min(OLD_IMAGES_SAMPLE, len(old_available)))
    state["sampled_old"] = sampled

    label_file = Path(state["label_file"])

    for img_name in sampled:
        text = find_label_in_history(img_name)
        if not text:
            try:
                text = gemini_extract_text(IMAGES_DIR / img_name)
            except Exception as e:
                print(f"[Label] Gemini failed for {img_name}: {e}")
                text = "unknown"
        append_label(label_file, img_name, text)
        state["total_labels_written"] += 1

    save_state(state)
    await run_training(state)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

def load_latest_ocr_model():
    """
    Load latest trained PaddleOCR model for inference.
    """

    global OCR_MODEL
    global OCR_MODEL_VERSION

    trained_versions = sorted(
        [p for p in MODELS_DIR.glob("v*/") if p.is_dir()],
        key=lambda p: int(p.name.replace("v", ""))
    )

    if not trained_versions:
        return None

    latest = trained_versions[-1]
    latest_version = latest.name

    # already loaded
    if OCR_MODEL and OCR_MODEL_VERSION == latest_version:
        return OCR_MODEL

    infer_model_dir = latest / "inference"

    if not infer_model_dir.exists():
        raise Exception(
            f"Inference model not found: {infer_model_dir}"
        )

    print(f"[OCR] Loading model: {latest_version}")

    OCR_MODEL = PaddleOCR(
        use_angle_cls=False,
        # use_gpu=False, 

        # disable default models
        det_model_dir=None,
        cls_model_dir=None,

        # your custom recognition model
        rec_model_dir=str(infer_model_dir),

        # IMPORTANT
        # rec_algorithm="SVTR_LCNet",
    )

    OCR_MODEL_VERSION = latest_version

    return OCR_MODEL
    
@app.post("/upload")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload one image at a time.
    - Saves to data/images/
    - Runs OpenCV preprocessing -> _gray.png + _otsu.png in data/processed/
    - Runs Gemini -> single-line text -> appended to label_vN.txt
    - After NEW_IMAGES_PER_SESSION images: samples old + triggers training
    - Returns 429 if training is running
    """
    state = load_state()

    # Block during training
    if state["status"] == "training":
        raise HTTPException(
            status_code=429,
            detail={
                "error":      "Training in progress. Wait until complete before uploading.",
                "status":     state["status"],
                "session_id": state["session_id"],
            },
        )

    # Block if threshold already hit
    if (state["status"] == "collecting"
            and len(state["new_images"]) >= NEW_IMAGES_PER_SESSION):
        raise HTTPException(
            status_code=429,
            detail={
                "error":  f"Already have {NEW_IMAGES_PER_SESSION} images. Training starting soon.",
                "status": state["status"],
            },
        )

    # Start new session
    if state["status"] in ("idle", "done", "error"):
        version    = next_version()
        session_id = f"session_{version}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        state      = _default_state()
        label_path = (LABELS_DIR / f"label_{version}.txt").resolve()
        state.update({
            "session_id":    session_id,
            "status":        "collecting",
            "model_version": version,
            "label_file":    str(label_path),   # absolute path — no double-prefix
        })
        label_path.touch()
        save_state(state)

    # Save image
    suffix   = Path(file.filename or "image.jpg").suffix.lower() or ".jpg"
    img_name = f"{state['session_id']}_{len(state['new_images']):03d}{suffix}"
    img_path = IMAGES_DIR / img_name
    img_path.write_bytes(await file.read())

    # OpenCV preprocessing -> _gray.png + _otsu.png
    proc_result = preprocess_and_save(img_path)

    # Gemini extraction (single line enforced)
    try:
        text = gemini_extract_text(img_path)
    except Exception as e:
        img_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Gemini extraction failed: {e}")

    # Append label (single line, absolute path)
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
        "message":          f"Image {collected}/{NEW_IMAGES_PER_SESSION} collected and labeled.",
    }

    # Auto-trigger training
    if collected >= NEW_IMAGES_PER_SESSION:
        response["message"] = (
            f"Threshold reached! Sampling {OLD_IMAGES_SAMPLE} old images "
            f"and starting training automatically."
        )
        background_tasks.add_task(prepare_and_train, state)

    return JSONResponse(content=response)


@app.get("/status")
async def get_status():
    """
    Return ALL sessions + system info.
    """

    sessions = load_all_sessions()

    trained = sorted(
        [p.parent.name for p in MODELS_DIR.glob("v*/best_accuracy.pdparams")],
        key=lambda x: int(x.replace("v", "")) if x.replace("v", "").isdigit() else 0,
    )

    label_files = sorted(
        [p.name for p in LABELS_DIR.glob("label_v*.txt")]
    )

    total_images = len([
        p for p in IMAGES_DIR.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ])

    total_proc = len(
        list(PROCESSED_DIR.glob("*_gray.png"))
    )

    prev_ckpt = find_previous_checkpoint()

    return {
        "total_sessions": len(sessions),

        "sessions": sessions,

        "incremental_chain": {
            "trained_versions": trained,
            "next_version": f"v{len(trained) + 1}",
            "prev_checkpoint":
                str(prev_ckpt) + ".pdparams"
                if prev_ckpt else None,
            "chain":
                " -> ".join(
                    trained + [f"v{len(trained)+1}(next)"]
                ),
        },

        "storage": {
            "total_images": total_images,
            "processed_images": total_proc,
            "label_files": label_files,
            "models_dir": str(MODELS_DIR),
            "processed_dir": str(PROCESSED_DIR),
        },

        "thresholds": {
            "new_images_per_session": NEW_IMAGES_PER_SESSION,
            "old_images_sampled": OLD_IMAGES_SAMPLE,
            "total_before_training": TOTAL_BEFORE_TRAIN,
        },
    }


@app.get("/status/{session_id}")
async def get_session_status(session_id: str):
    """Detailed status for a specific session including progress."""
    state = load_state(session_id)
    # if state.get("session_id") != session_id:
    #     raise HTTPException(
    #         status_code=404,
    #         detail=f"Session '{session_id}' not found. Current: {state.get('session_id')}",
    #     )

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
                "training"           if state["status"] == "training"
                else "waiting"       if collected < NEW_IMAGES_PER_SESSION
                else "starting_soon"
            ),
            "images_collected":  collected,
            "images_needed":     NEW_IMAGES_PER_SESSION,
            "images_remaining":  max(0, NEW_IMAGES_PER_SESSION - collected),
            "percent_collected": round(collected / NEW_IMAGES_PER_SESSION * 100, 1),
        },
    }


@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text using latest trained PaddleOCR model ONLY.
    """

    suffix = Path(file.filename or "img.jpg").suffix.lower() or ".jpg"
    tmp_path = DATA_DIR / f"_tmp_extract{suffix}"

    try:
        # save temp image
        tmp_path.write_bytes(await file.read())

        # preprocess image
        proc_result = preprocess_and_save(tmp_path)

        infer_image = proc_result.get("gray")

        if not infer_image:
            raise Exception("Preprocessing failed")

        # load latest trained OCR model
        ocr = load_latest_ocr_model()

        if ocr is None:
            raise Exception("No trained OCR model found")

        # OCR inference
        result = ocr.ocr(infer_image, det=False)

        extracted_text = ""

        try:
            if result and result[0]:
                extracted_text = result[0][0][0]
        except Exception:
            pass

        extracted_text = extracted_text.strip()

        return {
            "filename": file.filename,
            "model_used": OCR_MODEL_VERSION,
            "extracted_text": extracted_text,
            "char_count": len(extracted_text),
            "preprocessing": proc_result,
            "note": "Using trained PaddleOCR model only.",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR extraction failed: {e}"
        )

    finally:
        tmp_path.unlink(missing_ok=True)

        for p in DATA_DIR.glob("_tmp_extract*"):
            p.unlink(missing_ok=True)

        for p in PROCESSED_DIR.glob("_tmp_extract*"):
            p.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
