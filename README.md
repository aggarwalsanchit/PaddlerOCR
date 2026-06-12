# PaddleOCR Incremental Training Server

FastAPI server for incremental PaddleOCR training on digital displays.  
Same library versions as the working Colab notebook.

---

## Directory Structure

```
paddleocr_server/
├── main.py             ← FastAPI app (all 5 endpoints)
├── trainer.py          ← Incremental training logic
├── preprocessor.py     ← OpenCV pipeline (gray + Otsu)
├── label_store.py      ← Cumulative label merge
├── training_state.py   ← Thread-safe session tracker
├── config.py           ← All paths and settings
├── requirements.txt
└── data/
    ├── digital_dict.txt   ← PUT THIS HERE MANUALLY
    ├── images/            ← uploaded training images
    ├── labels/            ← uploaded label files + cumulative store
    ├── checkpoints/       ← v1.pdparams, v2.pdparams, ...
    └── processed/         ← gray + otsu preview images
```

---

## Server Setup

### 1. Install Python 3.12
```bash
python3 --version   # confirm 3.12.x
```

### 2. Create virtualenv
```bash
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
```

### 3. Install PaddlePaddle first (not on PyPI for Python 3.12)
```bash
# CPU server:
pip install https://paddle-whl.bj.bcebos.com/stable/linux/cpu-mkl-avx/paddlepaddle-2.6.2-cp312-cp312-linux_x86_64.whl

# GPU server (CUDA 11.8):
pip install https://paddle-whl.bj.bcebos.com/stable/linux/gpu-cuda11.8-cudnn8.6-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.2.post118-cp312-cp312-linux_x86_64.whl
```

### 4. Install remaining packages
```bash
pip install -r requirements.txt
```

### 5. Add your dictionary file
```bash
cp /your/path/digital_dict.txt data/digital_dict.txt
```

### 6. Start the server
```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for Swagger UI.

---

## 5 APIs

### API 1 — Upload Images
```
POST /upload-images
```
Upload one or more image files (jpg/png/bmp).
```bash
curl -X POST http://localhost:8000/upload-images \
  -F "files=@img001.jpg" \
  -F "files=@img002.jpg"
```
Response:
```json
{
  "saved": ["img001.jpg", "img002.jpg"],
  "saved_count": 2,
  "total_images_on_server": 52
}
```

---

### API 2 — Upload Labels → Starts Training
```
POST /upload-labels
```
Upload a label .txt file. Training starts **automatically** in the background.
```bash
curl -X POST http://localhost:8000/upload-labels \
  -F "file=@labels_day1.txt"
```
Response:
```json
{
  "session_id": "s1716200000",
  "new_labels": 50,
  "total_cumulative": 100,
  "status": "training_started",
  "poll_url": "/status/s1716200000"
}
```

---

### API 3 — Extract (OCR Inference)
```
POST /extract
```
Upload any image, get the OCR result using the latest trained model.
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@display_photo.jpg"
```
Response:
```json
{
  "best_result": "23.5°C",
  "confidence": 0.9823,
  "variants": {
    "original": [{"text": "23.5°C", "confidence": 0.9823}],
    "gray":     [{"text": "23.5°C", "confidence": 0.9751}],
    "otsu":     [{"text": "23.5°C", "confidence": 0.9812}]
  },
  "checkpoint": "v2.pdparams"
}
```

---

### API 4 — Full Status & History
```
GET /status
```
Returns everything: all sessions, current training progress, checkpoint history, live log.
```bash
curl http://localhost:8000/status
```
Response includes:
```json
{
  "currently_running": {
    "status": "training",
    "progress_pct": 64,
    "current_epoch": 32,
    "total_epochs": 50
  },
  "live_log": ["epoch: [32/50] ...", "..."],
  "checkpoint_history": [
    {"name": "v1.pdparams", "size_mb": 2.5},
    {"name": "v2.pdparams", "size_mb": 2.5}
  ],
  "sessions": [...]
}
```

---

### API 5 — Single Session Detail
```
GET /status/{session_id}
```
Full detail for one session with log tail.
```bash
curl http://localhost:8000/status/s1716200000?log_lines=200
```

---

## Incremental Training — How It Works

```
Upload 50 images + labels_day1.txt  →  trains on 50  →  saves v1.pdparams
Upload 50 images + labels_day2.txt  →  trains on 100 →  saves v2.pdparams
Upload 50 images + labels_day3.txt  →  trains on 150 →  saves v3.pdparams
```

The cumulative label store is at `data/labels/cumulative_raw.json`.  
**Never delete this file** — it's the memory of all previous sessions.

---

## Label File Format

Tab-separated, one entry per line:
```
image001.jpg	23.5°C
image002.jpg	1234
image003.jpg	56.7V
```
Lines starting with `#` are ignored.

---

## Configuration (config.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `EPOCHS` | 50 | Epochs per session |
| `BATCH_SIZE` | 8 | Lower to 4 if OOM |
| `LEARNING_RATE` | 0.001 | Initial LR |
| `USE_GPU` | False | Set True for CUDA |
| `IMAGE_SHAPE` | [3,48,320] | PaddleOCR input |

---

## Deployment with systemd

```ini
# /etc/systemd/system/paddleocr.service
[Unit]
Description=PaddleOCR Training Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/paddleocr_server
ExecStart=/home/ubuntu/paddleocr_server/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable paddleocr
sudo systemctl start paddleocr
```

---

## Common Errors

**`ModuleNotFoundError: No module named 'paddle'`**  
PaddlePaddle is not on PyPI for Python 3.12. Install via the wheel URL in step 3 above.

**`digital_dict.txt not found`**  
Copy your dict file to `data/digital_dict.txt` on the server.

**`409 A training session is already running`**  
Wait for the current session to finish. Check `GET /status`.

**`404 No trained checkpoint found`**  
Train at least one session before calling `/extract`.
# PaddlerOCR
