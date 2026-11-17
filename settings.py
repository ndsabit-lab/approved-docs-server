"""Shared settings and config loading for the archive server."""
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data")).resolve()
TMP_DIR = DATA_DIR / "tmp"
DB_PATH = DATA_DIR / "archive.db"
API_KEY = os.environ.get("ARCHIVE_API_KEY", "")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "40")) * 1024 * 1024

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _load(name: str) -> dict:
    with open(BASE_DIR / "config" / name, encoding="utf-8") as f:
        return json.load(f)


DOC_TYPES = _load("doc_types.json")["doc_types"]
LABELS = [t["value"] for t in DOC_TYPES]
TYPE_FOLDERS = {t["value"]: t["folder"] for t in DOC_TYPES}
OFFICES = _load("offices.json")["offices"]
