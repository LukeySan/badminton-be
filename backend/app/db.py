"""
SQLite access shared by the API and (from milestone 3) the worker.

Status changes are single conditional UPDATEs ("... WHERE status IN (...)"), so the
check and the write happen atomically: two clicks on "process", or two workers
claiming jobs, can't both succeed.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,             -- the user's original name, display only
    storage_path  TEXT NOT NULL,             -- relative to STORAGE_DIR
    status        TEXT NOT NULL DEFAULT 'uploaded',
    progress      REAL NOT NULL DEFAULT 0,   -- 0-1 while processing
    error         TEXT,
    duration      REAL,
    fps           REAL,
    width         INTEGER,                   -- as displayed (rotation applied)
    height        INTEGER,
    court_corners TEXT,                      -- JSON [[x,y] x4] fractions 0-1
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL              -- worker refreshes it as a heartbeat
);

CREATE TABLE IF NOT EXISTS rallies (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id   TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start      REAL NOT NULL,
    "end"      REAL NOT NULL,
    source     TEXT NOT NULL DEFAULT 'rules',  -- 'rules' | 'model'
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS rallies_video ON rallies(video_id, start);

-- Not used yet: user edits to rallies, kept with the detector's original as training data.
CREATE TABLE IF NOT EXISTS corrections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    rally_id       INTEGER,
    video_id       TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    original_start REAL,
    original_end   REAL,
    edited_start   REAL,
    edited_end     REAL,
    action         TEXT NOT NULL,              -- 'edit' | 'delete' | 'add'
    created_at     TEXT NOT NULL
);
"""

# Where each status may move to. "Queued" is allowed again from done/failed so a
# video can be retried or reprocessed after fixing the court.
CAN_QUEUE_FROM = ("uploaded", "done", "failed")
CAN_EDIT_COURT_IN = ("uploaded", "done", "failed")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    # FastAPI may run a request's dependency and its handler on different threads.
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")  # API reads don't block the worker's writes
    return conn


def init(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def video_to_dict(row):
    d = dict(row)
    d.pop("storage_path")
    d["court_corners"] = json.loads(d["court_corners"]) if d["court_corners"] else None
    return d


def get_video(conn, video_id):
    return conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()


def list_videos(conn):
    return conn.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()


def insert_video(conn, video_id, filename, storage_path, info):
    t = now()
    conn.execute(
        "INSERT INTO videos (id, filename, storage_path, duration, fps, width, height, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (video_id, filename, storage_path, info["duration"], info["fps"], info["width"], info["height"], t, t),
    )
    conn.commit()


def set_court(conn, video_id, corners):
    """Returns False if the video is queued/processing (or doesn't exist)."""
    placeholders = ",".join("?" * len(CAN_EDIT_COURT_IN))
    cur = conn.execute(
        f"UPDATE videos SET court_corners = ?, updated_at = ? WHERE id = ? AND status IN ({placeholders})",
        (json.dumps(corners), now(), video_id, *CAN_EDIT_COURT_IN),
    )
    conn.commit()
    return cur.rowcount == 1


def queue(conn, video_id):
    """Returns False if the video can't be queued right now (or doesn't exist)."""
    placeholders = ",".join("?" * len(CAN_QUEUE_FROM))
    cur = conn.execute(
        "UPDATE videos SET status = 'queued', progress = 0, error = NULL, updated_at = ? "
        f"WHERE id = ? AND court_corners IS NOT NULL AND status IN ({placeholders})",
        (now(), video_id, *CAN_QUEUE_FROM),
    )
    conn.commit()
    return cur.rowcount == 1


def get_rallies(conn, video_id):
    return conn.execute(
        'SELECT id, start, "end" FROM rallies WHERE video_id = ? ORDER BY start', (video_id,)
    ).fetchall()


def new_id():
    return str(uuid.uuid4())
