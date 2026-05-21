"""
label_store.py
Parses uploaded label files and maintains the cumulative label store.
"""
import os, json
from typing import List, Tuple
from config import CUMULATIVE_JSON, CUMULATIVE_LABEL_TXT, IMAGES_DIR


def parse_label_file(path: str) -> List[Tuple[str, str]]:
    """
    Parse a label .txt file.
    Supported formats:
        image.jpg<TAB>text
        image.jpg text
    Lines starting with # are ignored.
    Returns list of (basename, text) tuples.
    """
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            fname = os.path.basename(parts[0].strip())
            text  = parts[1].strip().strip('"').strip("'")
            if fname and text:
                entries.append((fname, text))
    return entries


def load_cumulative() -> dict:
    """Return {filename: text} from the cumulative store."""
    if os.path.exists(CUMULATIVE_JSON):
        try:
            with open(CUMULATIVE_JSON) as f:
                return {k: v for k, v in json.load(f)}
        except Exception:
            pass
    return {}


def save_cumulative(store: dict):
    os.makedirs(os.path.dirname(CUMULATIVE_JSON), exist_ok=True)
    with open(CUMULATIVE_JSON, "w") as f:
        json.dump(list(store.items()), f, indent=2)


def merge_and_write(new_entries: List[Tuple[str, str]]) -> Tuple[dict, int, int]:
    """
    Merge new_entries into the cumulative store.
    Writes the PaddleOCR-format merged label file.
    Returns (merged_store, written_count, skipped_count)
    """
    store  = load_cumulative()
    before = len(store)
    for fname, text in new_entries:
        store[fname] = text
    save_cumulative(store)

    # Write PaddleOCR label file: full_image_path TAB text
    written = skipped = 0
    with open(CUMULATIVE_LABEL_TXT, "w", encoding="utf-8") as f:
        for fname, text in store.items():
            img_path = os.path.join(IMAGES_DIR, fname)
            if os.path.exists(img_path):
                f.write(f"{img_path}\t{text}\n")
                written += 1
            else:
                skipped += 1

    return store, written, skipped


def cumulative_count() -> int:
    return len(load_cumulative())
