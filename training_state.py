"""
training_state.py
Thread-safe singleton that tracks training sessions and progress.
Written to disk (sessions.json) so it survives server restarts.
"""
import os, json, time, threading
from typing import Optional
from config import LOGS_DIR, DATA_DIR

SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")

_lock = threading.Lock()

# ── In-memory state ───────────────────────────────────────────────────────────
_sessions: list  = []          # list of session dicts
_current_log: list = []        # live log lines for the running session


def _load():
    global _sessions
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE) as f:
                _sessions = json.load(f)
        except Exception:
            _sessions = []


def _save():
    with open(SESSIONS_FILE, "w") as f:
        json.dump(_sessions, f, indent=2)


_load()


# ── Public API ────────────────────────────────────────────────────────────────

def create_session(label_file: str, image_count: int, total_cumulative: int) -> str:
    with _lock:
        sid = f"s{int(time.time())}"
        session = {
            "session_id":        sid,
            "started_at":        time.strftime("%Y-%m-%dT%H:%M:%S"),
            "finished_at":       None,
            "status":            "queued",   # queued|preprocessing|training|done|error
            "label_file":        label_file,
            "new_images":        image_count,
            "total_images":      total_cumulative,
            "checkpoint_path":   None,
            "error":             None,
            "progress_pct":      0,
            "current_epoch":     0,
            "total_epochs":      0,
            "log_file":          os.path.join(LOGS_DIR, f"{sid}.log"),
        }
        _sessions.append(session)
        _save()
        return sid


def update(sid: str, **kwargs):
    with _lock:
        for s in _sessions:
            if s["session_id"] == sid:
                s.update(kwargs)
                _save()
                return


def get_session(sid: str) -> Optional[dict]:
    for s in _sessions:
        if s["session_id"] == sid:
            return dict(s)
    return None


def get_all_sessions() -> list:
    return list(reversed(_sessions))


def latest_checkpoint() -> Optional[str]:
    for s in reversed(_sessions):
        if s["status"] == "done" and s.get("checkpoint_path"):
            if os.path.exists(s["checkpoint_path"]):
                return s["checkpoint_path"]
    return None


def is_running() -> bool:
    return any(s["status"] in ("queued", "preprocessing", "training")
               for s in _sessions)


def total_trained_images() -> int:
    done = [s for s in _sessions if s["status"] == "done"]
    return done[-1]["total_images"] if done else 0


# ── Live log buffer ───────────────────────────────────────────────────────────

def log_line(line: str):
    with _lock:
        _current_log.append(line)
        if len(_current_log) > 500:
            _current_log.pop(0)


def get_live_log(last_n: int = 100) -> list:
    return _current_log[-last_n:]


def clear_live_log():
    with _lock:
        _current_log.clear()
