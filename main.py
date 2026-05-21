"""
main.py  —  PaddleOCR Incremental Training Server
5 endpoints:
  POST /upload-images          Upload training images
  POST /upload-labels          Upload label txt → triggers preprocessing + training
  POST /extract                OCR inference on an uploaded image
  GET  /status                 Full training history, current progress, stats
  GET  /status/{session_id}    Single session detail + live log tail
"""

import os, time, glob, shutil, threading, tempfile
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import config
import training_state as state
from label_store   import parse_label_file, merge_and_write, cumulative_count
from preprocessor  import preprocess_image
from trainer       import run_training, list_checkpoints, find_latest_checkpoint


# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("PaddleOCR Server started")
    print(f"  Images dir  : {config.IMAGES_DIR}")
    print(f"  Dict file   : {config.DICT_FILE}")
    print(f"  Dict exists : {os.path.exists(config.DICT_FILE)}")
    yield
    print("PaddleOCR Server stopped")


app = FastAPI(
    title       = "PaddleOCR Incremental Training Server",
    description = "Train PaddleOCR incrementally on digital displays. "
                  "Place digital_dict.txt in data/ before use.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_train_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
#  API 1 — Upload images
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/upload-images",
    summary="① Upload training images",
    tags=["Training"],
)
async def upload_images(files: List[UploadFile] = File(...)):
    """
    Upload one or more image files (jpg / png / bmp).
    Saved to  data/images/  ready for the next training session.
    """
    if not files:
        raise HTTPException(400, "No files provided")

    saved, failed = [], []
    for upload in files:
        fname = upload.filename or f"img_{int(time.time())}.jpg"
        if not any(fname.lower().endswith(e)
                   for e in (".jpg", ".jpeg", ".png", ".bmp")):
            failed.append({"file": fname, "error": "Unsupported extension"})
            continue
        dest = os.path.join(config.IMAGES_DIR, fname)
        try:
            content = await upload.read()
            with open(dest, "wb") as f:
                f.write(content)
            saved.append(fname)
        except Exception as e:
            failed.append({"file": fname, "error": str(e)})

    total_images = len([
        f for f in os.listdir(config.IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    return {
        "saved":              saved,
        "failed":             failed,
        "saved_count":        len(saved),
        "total_images_on_server": total_images,
        "message": f"{len(saved)} image(s) uploaded successfully.",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  API 2 — Upload label file → triggers training
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/upload-labels",
    summary="② Upload label file → starts preprocessing + training",
    tags=["Training"],
)
async def upload_labels(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a label .txt file.
    Format (tab-separated, one line per image):
        image001.jpg<TAB>1234
        image002.jpg<TAB>56.7°C

    After saving, preprocessing + incremental training start automatically.
    Poll  GET /status  or  GET /status/{session_id}  to track progress.
    """
    # ── Guards ────────────────────────────────────────────────────────────────
    if state.is_running():
        raise HTTPException(
            409,
            "A training session is already running. "
            "Wait for it to finish before uploading new labels.",
        )
    if not os.path.exists(config.DICT_FILE):
        raise HTTPException(
            400,
            f"digital_dict.txt not found at {config.DICT_FILE}. "
            "Place it in the data/ directory on the server.",
        )
    img_count = len([
        f for f in os.listdir(config.IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])
    if img_count == 0:
        raise HTTPException(
            400,
            "No images in data/images/. Upload images first via POST /upload-images.",
        )

    # ── Save label file ───────────────────────────────────────────────────────
    os.makedirs(config.LABELS_DIR, exist_ok=True)
    fname      = file.filename or f"labels_{int(time.time())}.txt"
    label_dest = os.path.join(
        config.LABELS_DIR, f"{int(time.time())}_{fname}"
    )
    content = await file.read()
    with open(label_dest, "wb") as f:
        f.write(content)

    # ── Parse ─────────────────────────────────────────────────────────────────
    try:
        new_entries = parse_label_file(label_dest)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse label file: {e}")
    if not new_entries:
        raise HTTPException(400, "Label file is empty or has no valid entries.")

    # ── Merge into cumulative store ───────────────────────────────────────────
    store, written, skipped = merge_and_write(new_entries)
    total_cumulative         = len(store)

    # ── Create session & launch ───────────────────────────────────────────────
    sid = state.create_session(label_dest, len(new_entries), total_cumulative)
    state.clear_live_log()
    background_tasks.add_task(_train_background, sid)

    return {
        "session_id":        sid,
        "new_labels":        len(new_entries),
        "total_cumulative":  total_cumulative,
        "label_entries_written": written,
        "label_entries_skipped": skipped,
        "status":            "training_started",
        "poll_url":          f"/status/{sid}",
        "message": (
            f"Session {sid} started. "
            f"{len(new_entries)} new + {total_cumulative - len(new_entries)} existing "
            f"= {total_cumulative} total training images."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  API 3 — Extract / Inference
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/extract",
    summary="③ OCR extraction from image using latest trained model",
    tags=["Inference"],
)
async def extract(file: UploadFile = File(...)):
    """
    Upload an image and get the OCR text result.

    Runs inference on three variants:
    - original image
    - gray-enhanced (CLAHE)
    - Otsu binary

    Returns the result from each variant + a best_result pick.
    Uses the latest trained checkpoint automatically.
    """
    if not os.path.exists(config.DICT_FILE):
        raise HTTPException(
            400, "digital_dict.txt not found. Cannot run inference."
        )

    checkpoint = find_latest_checkpoint()
    if not checkpoint:
        raise HTTPException(
            404,
            "No trained checkpoint found. "
            "Train the model first via POST /upload-labels.",
        )

    # Save uploaded image to temp file
    suffix = os.path.splitext(file.filename or "img.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Load OCR model (lazy — cached after first call)
        ocr = _get_ocr(checkpoint)

        # Preprocess
        original, (gray, otsu) = preprocess_image(tmp_path)

        def _run_on_array(arr):
            import cv2
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
                cv2.imwrite(t.name, arr)
                r = ocr.ocr(t.name, cls=False)
                os.unlink(t.name)
                return r

        results = {}
        for variant, img_or_path in [
            ("original", tmp_path),
            ("gray",     gray),
            ("otsu",     otsu),
        ]:
            try:
                if isinstance(img_or_path, str):
                    raw = ocr.ocr(img_or_path, cls=False)
                else:
                    raw = _run_on_array(img_or_path)
                if raw and raw[0]:
                    results[variant] = [
                        {"text": t, "confidence": round(float(c), 4)}
                        for _, (t, c) in raw[0]
                    ]
                else:
                    results[variant] = []
            except Exception as e:
                results[variant] = {"error": str(e)}

        # Pick best result: highest confidence across variants
        best_text = None
        best_conf = 0.0
        for variant_res in results.values():
            if isinstance(variant_res, list):
                for item in variant_res:
                    if isinstance(item, dict) and item.get("confidence", 0) > best_conf:
                        best_conf = item["confidence"]
                        best_text = item["text"]

        return {
            "best_result":  best_text,
            "confidence":   best_conf,
            "variants":     results,
            "checkpoint":   os.path.basename(checkpoint),
            "image":        file.filename,
        }

    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
#  API 4 — Full training history + status
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/status",
    summary="④ Full training history, progress and stats",
    tags=["Status"],
)
def get_status():
    """
    Returns:
    - All training sessions (newest first)
    - Currently running session progress (epoch, %)
    - Checkpoint history with sizes
    - Overall stats (total images, sessions, latest checkpoint)
    - Live log (last 50 lines of the running session)
    """
    sessions     = state.get_all_sessions()
    running      = next((s for s in sessions
                         if s["status"] in ("queued","preprocessing","training")),
                        None)
    latest_ckpt  = find_latest_checkpoint()
    ckpt_history = list_checkpoints()

    return {
        "service":           "PaddleOCR Incremental Training Server",
        "dict_ready":        os.path.exists(config.DICT_FILE),
        "images_on_server":  len([
            f for f in os.listdir(config.IMAGES_DIR)
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))
        ]),
        "total_sessions":    len(sessions),
        "total_trained_images": state.total_trained_images(),
        "latest_checkpoint": os.path.basename(latest_ckpt) if latest_ckpt else None,

        "currently_running": {
            "session_id":    running["session_id"]   if running else None,
            "status":        running["status"]        if running else "idle",
            "progress_pct":  running["progress_pct"] if running else 0,
            "current_epoch": running["current_epoch"] if running else 0,
            "total_epochs":  running["total_epochs"] if running else 0,
        },

        "live_log": state.get_live_log(50) if running else [],

        "checkpoint_history": ckpt_history,

        "sessions": [
            {
                "session_id":      s["session_id"],
                "status":          s["status"],
                "started_at":      s["started_at"],
                "finished_at":     s["finished_at"],
                "new_images":      s["new_images"],
                "total_images":    s["total_images"],
                "progress_pct":    s["progress_pct"],
                "checkpoint":      os.path.basename(s["checkpoint_path"])
                                   if s.get("checkpoint_path") else None,
                "error":           s["error"],
            }
            for s in sessions
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  API 5 — Single session detail
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/status/{session_id}",
    summary="⑤ Single session detail + live log",
    tags=["Status"],
)
def get_session_status(
    session_id: str,
    log_lines: int = Query(100, ge=10, le=1000,
                           description="How many log lines to return"),
):
    """
    Returns full detail for one session including the tail of its log file.
    """
    session = state.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found")

    # Read log file
    log_tail = []
    log_file = session.get("log_file", "")
    if log_file and os.path.exists(log_file):
        with open(log_file, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        log_tail = [l.rstrip() for l in lines[-log_lines:]]

    # Is this the currently running session?
    is_live = session["status"] in ("queued", "preprocessing", "training")

    return {
        **session,
        "checkpoint_name": os.path.basename(session["checkpoint_path"])
                           if session.get("checkpoint_path") else None,
        "log_tail":  log_tail,
        "live_log":  state.get_live_log(50) if is_live else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  OCR model cache  (reload only when checkpoint changes)
# ─────────────────────────────────────────────────────────────────────────────

_ocr_cache      = {"ckpt": None, "model": None}
_ocr_lock       = threading.Lock()


def _get_ocr(checkpoint: str):
    with _ocr_lock:
        if _ocr_cache["ckpt"] != checkpoint:
            from paddleocr import PaddleOCR
            _ocr_cache["model"] = PaddleOCR(
                use_angle_cls      = False,
                rec_model_dir      = os.path.dirname(checkpoint),
                rec_char_dict_path = config.DICT_FILE,
                use_gpu            = config.USE_GPU,
                det                = False,
                show_log           = False,
            )
            _ocr_cache["ckpt"] = checkpoint
        return _ocr_cache["model"]


# ─────────────────────────────────────────────────────────────────────────────
#  Background training runner
# ─────────────────────────────────────────────────────────────────────────────

def _train_background(sid: str):
    with _train_lock:
        run_training(sid)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
