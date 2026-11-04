import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Override with env vars to point the API (and later the worker) somewhere else.
STORAGE_DIR = Path(os.environ.get("BADMINTON_STORAGE", REPO_ROOT / "storage"))
DB_PATH = Path(os.environ.get("BADMINTON_DB", REPO_ROOT / "data" / "app.db"))

CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
