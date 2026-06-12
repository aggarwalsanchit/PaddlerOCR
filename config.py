import os

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
IMAGES_DIR      = os.path.join(DATA_DIR, "images")
LABELS_DIR      = os.path.join(DATA_DIR, "labels")
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
WORK_DIR        = os.path.join(BASE_DIR, "work")

# ── Dictionary file (place digital_dict.txt in data/ manually on server) ─────
DICT_FILE            = os.path.join(DATA_DIR, "digital_dict.txt")

# ── Cumulative label store ────────────────────────────────────────────────────
CUMULATIVE_JSON      = os.path.join(LABELS_DIR, "cumulative_raw.json")
CUMULATIVE_LABEL_TXT = os.path.join(LABELS_DIR, "cumulative_labels.txt")

# ── Training settings (match notebook) ───────────────────────────────────────
EPOCHS         = 50
BATCH_SIZE     = 8
LEARNING_RATE  = 0.001
IMAGE_SHAPE    = [3, 48, 320]
USE_GPU        = False      # set True if server has CUDA GPU

# ── OpenCV preprocessing (exact values from your image_processing.py) ────────
UPSCALE_FACTOR  = 4
BILATERAL_D     = 9
BILATERAL_SC    = 75
BILATERAL_SS    = 75
CLAHE_CLIP      = 3.0
CLAHE_TILE      = (8, 8)
MIN_AREA        = 1000
MIN_W           = 80
MIN_H           = 40
PAD             = 20

# ── Create all dirs on import ─────────────────────────────────────────────────
for _d in [IMAGES_DIR, LABELS_DIR, CHECKPOINTS_DIR,
           PROCESSED_DIR, LOGS_DIR, WORK_DIR]:
    os.makedirs(_d, exist_ok=True)
