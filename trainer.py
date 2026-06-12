"""
trainer.py
Runs one incremental PaddleOCR training session.
Picks up from the latest vN.pdparams checkpoint automatically.
Saves the new checkpoint as v(N+1).pdparams.
"""
import os, re, glob, shutil, time, subprocess, sys, yaml, logging

import training_state as state
from config import (
    CHECKPOINTS_DIR, DICT_FILE, WORK_DIR, LOGS_DIR,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, IMAGE_SHAPE,
    IMAGES_DIR, USE_GPU,
)
from label_store import CUMULATIVE_LABEL_TXT


# ── Checkpoint versioning ─────────────────────────────────────────────────────

def find_latest_checkpoint() -> str | None:
    files = glob.glob(os.path.join(CHECKPOINTS_DIR, "v*.pdparams"))
    if not files:
        return None
    files.sort(key=lambda p: int(re.search(r"v(\d+)\.pdparams", p).group(1)))
    return files[-1]


def next_checkpoint_path() -> str:
    latest = find_latest_checkpoint()
    if latest is None:
        return os.path.join(CHECKPOINTS_DIR, "v1.pdparams")
    n = int(re.search(r"v(\d+)\.pdparams", latest).group(1)) + 1
    return os.path.join(CHECKPOINTS_DIR, f"v{n}.pdparams")


def list_checkpoints() -> list:
    files = glob.glob(os.path.join(CHECKPOINTS_DIR, "v*.pdparams"))
    files.sort(key=lambda p: int(re.search(r"v(\d+)\.pdparams", p).group(1)))
    result = []
    for p in files:
        result.append({
            "name":       os.path.basename(p),
            "path":       p,
            "size_mb":    round(os.path.getsize(p) / (1024 * 1024), 2),
            "created_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S",
                time.localtime(os.path.getmtime(p))
            ),
        })
    return result


# ── Session-scoped file logger ────────────────────────────────────────────────

def _make_logger(sid: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(sid)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    logger.addHandler(fh)
    return logger


def _log(sid: str, logger, msg: str):
    logger.info(msg)
    state.log_line(f"[{sid}] {msg}")


# ── Main training entry point ─────────────────────────────────────────────────

def run_training(sid: str):
    """
    Called in a background thread.
    Full lifecycle: preprocess → build config → clone repo → train → save checkpoint.
    """
    session  = state.get_session(sid)
    log_file = session["log_file"]
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger   = _make_logger(sid, log_file)

    try:
        # ── 1. Preprocessing ─────────────────────────────────────────────────
        state.update(sid, status="preprocessing", progress_pct=5)
        _log(sid, logger, "Starting preprocessing...")

        from preprocessor import process_and_save
        import time as _time
        proc_dir = os.path.join(
            os.path.join(os.path.dirname(CHECKPOINTS_DIR), "processed"),
            f"session_{sid}"
        )
        img_paths = [
            os.path.join(IMAGES_DIR, f)
            for f in os.listdir(IMAGES_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        results  = process_and_save(img_paths, proc_dir)
        ok_count = sum(1 for v in results.values() if "error" not in v)
        _log(sid, logger, f"Preprocessed {ok_count}/{len(img_paths)} images")
        state.update(sid, progress_pct=15)

        # ── 2. Verify label file ──────────────────────────────────────────────
        if not os.path.exists(CUMULATIVE_LABEL_TXT):
            raise FileNotFoundError("Cumulative label file not found. "
                                    "Upload a label file first.")
        with open(CUMULATIVE_LABEL_TXT) as f:
            label_lines = [l for l in f if l.strip()]
        if not label_lines:
            raise ValueError("Label file is empty.")
        _log(sid, logger, f"Label file OK: {len(label_lines)} entries")

        # ── 3. Detect / download PaddleOCR repo ──────────────────────────────
        state.update(sid, status="training", progress_pct=20)
        repo_dir     = os.path.join(WORK_DIR, "PaddleOCR")
        train_script = os.path.join(repo_dir, "tools", "train.py")

        if not os.path.exists(train_script):
            _log(sid, logger, "Cloning PaddleOCR repo...")
            subprocess.run(
                ["/usr/bin/git", "clone", "--depth", "1",
                 "https://github.com/PaddlePaddle/PaddleOCR.git", repo_dir],
                check=True,
                capture_output=True,
            )
            _log(sid, logger, "PaddleOCR repo cloned OK")

        # ── 4. Resolve checkpoint ─────────────────────────────────────────────
        latest_ckpt  = find_latest_checkpoint()
        new_ckpt     = next_checkpoint_path()
        session_ckpt = os.path.join(WORK_DIR, f"ckpt_{sid}")
        os.makedirs(session_ckpt, exist_ok=True)

        if latest_ckpt:
            _log(sid, logger, f"Resuming from: {latest_ckpt}")
        else:
            _log(sid, logger, "No prior checkpoint — using pretrained backbone")

        # ── 5. Build YAML config ──────────────────────────────────────────────
        config = {
            "Architecture": {
                "model_type": "rec",
                "algorithm":  "CRNN",
                "Backbone": {"name": "MobileNetV3", "scale": 0.5,
                             "model_name": "small"},
                "Neck":     {"name": "SequenceEncoder",
                             "encoder_type": "rnn", "hidden_size": 48},
                "Head":     {"name": "CTCHead", "fc_decay": 4e-4},
            },
            "Global": {
                "use_gpu":                USE_GPU,
                "epoch_num":              EPOCHS,
                "log_smooth_window":      20,
                "print_batch_step":       10,
                "save_model_dir":         session_ckpt,
                "save_epoch_step":        max(1, EPOCHS // 5),
                "eval_batch_step":        [0, 200],
                "cal_metric_during_train": False,
                "pretrained_model":       latest_ckpt or "",
                "checkpoints":            None,
                "character_dict_path":    DICT_FILE,
                "max_text_length":        25,
                "infer_mode":             False,
                "use_space_char":         False,
            },
            "Loss":        {"name": "CTCLoss"},
            "PostProcess": {"name": "CTCLabelDecode"},
            "Metric":      {"name": "RecMetric", "main_indicator": "acc"},
            "Optimizer": {
                "name": "Adam",
                "lr":   {"name": "Cosine", "learning_rate": LEARNING_RATE},
                "regularizer": {"name": "L2", "factor": 4e-5},
            },
            "Train": {
                "dataset": {
                    "name":            "SimpleDataSet",
                    "data_dir":        IMAGES_DIR,
                    "label_file_list": [CUMULATIVE_LABEL_TXT],
                    "transforms": [
                        {"DecodeImage":    {"img_mode": "BGR",
                                           "channel_first": False}},
                        {"CTCLabelEncode": {}},
                        {"RecResizeImg":   {"image_shape": IMAGE_SHAPE}},
                        {"KeepKeys":       {"keep_keys":
                                           ["image", "label", "length"]}},
                    ],
                },
                "loader": {
                    "shuffle":             True,
                    "batch_size_per_card": BATCH_SIZE,
                    "drop_last":           True,
                    "num_workers":         2,
                },
            },
            "Eval": {
                "dataset": {
                    "name":            "SimpleDataSet",
                    "data_dir":        IMAGES_DIR,
                    "label_file_list": [CUMULATIVE_LABEL_TXT],
                    "transforms": [
                        {"DecodeImage":    {"img_mode": "BGR",
                                           "channel_first": False}},
                        {"CTCLabelEncode": {}},
                        {"RecResizeImg":   {"image_shape": IMAGE_SHAPE}},
                        {"KeepKeys":       {"keep_keys":
                                           ["image", "label", "length"]}},
                    ],
                },
                "loader": {
                    "shuffle":             False,
                    "batch_size_per_card": BATCH_SIZE,
                    "drop_last":           False,
                    "num_workers":         2,
                },
            },
        }

        cfg_path = os.path.join(WORK_DIR, f"config_{sid}.yml")
        with open(cfg_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        _log(sid, logger, f"Config written: {cfg_path}")

        # ── 6. Run training subprocess ────────────────────────────────────────
        state.update(sid, total_epochs=EPOCHS, progress_pct=25)
        _log(sid, logger, f"Training: {len(label_lines)} images, "
                          f"{EPOCHS} epochs, batch={BATCH_SIZE}")

        proc = subprocess.Popen(
            [sys.executable, train_script, "-c", cfg_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        epoch_re = re.compile(r"epoch:\s*\[?(\d+)/(\d+)")
        for line in proc.stdout:
            line = line.rstrip()
            logger.info(line)
            state.log_line(line)

            # Parse epoch progress from training output
            m = epoch_re.search(line)
            if m:
                cur, total = int(m.group(1)), int(m.group(2))
                pct = 25 + int((cur / total) * 70)
                state.update(sid, current_epoch=cur,
                             total_epochs=total, progress_pct=pct)

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Training process exited with code {proc.returncode}. "
                "Check logs for details."
            )

        # ── 7. Save versioned checkpoint to CHECKPOINTS_DIR ──────────────────
        state.update(sid, progress_pct=97)
        best = os.path.join(session_ckpt, "best_accuracy.pdparams")
        if not os.path.exists(best):
            # Fall back to latest epoch file
            candidates = sorted(
                glob.glob(os.path.join(session_ckpt, "*.pdparams")),
                key=os.path.getmtime,
            )
            if not candidates:
                raise FileNotFoundError(
                    "No .pdparams file found after training."
                )
            best = candidates[-1]

        shutil.copy2(best, new_ckpt)
        # Also save as latest.pdparams
        shutil.copy2(best, os.path.join(CHECKPOINTS_DIR, "latest.pdparams"))
        _log(sid, logger, f"Checkpoint saved: {new_ckpt}")

        state.update(
            sid,
            status          = "done",
            checkpoint_path = new_ckpt,
            finished_at     = time.strftime("%Y-%m-%dT%H:%M:%S"),
            progress_pct    = 100,
        )
        _log(sid, logger, "Training session complete.")

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        if "logger" in dir():
            logger.error(tb)
        state.log_line(f"[{sid}] ERROR: {exc}")
        state.update(
            sid,
            status      = "error",
            error       = str(exc),
            finished_at = time.strftime("%Y-%m-%dT%H:%M:%S"),
            progress_pct= 0,
        )
