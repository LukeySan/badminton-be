"""
A video's thumbnail strip (sprite sheets + index.json), written by
worker/thumbnail_strip.py into storage/<video id>/thumbnails/ (format described there;
keep the two in sync). The API only serves the files, through signed links.
"""

import json
import re
import uuid
from pathlib import Path
from typing import Any

from app.core.config import settings

DIRNAME = "thumbnails"
INDEX = "index.json"
# The only file names served: no paths, so a crafted name can't leave the folder
FILE_NAME = re.compile(r"^(index\.json|sheet_\d{3,}\.jpg)$")


def thumbnails_dir(video_id: uuid.UUID) -> Path:
    return settings.STORAGE_DIR / str(video_id) / DIRNAME


def read_index(video_id: uuid.UUID) -> dict[str, Any] | None:
    """None if the video has no thumbnails (analyzed before they existed, or not
    analyzed yet)."""
    try:
        index: dict[str, Any] = json.loads(
            (thumbnails_dir(video_id) / INDEX).read_text()
        )
    except FileNotFoundError:
        return None
    return index


def file_path(video_id: uuid.UUID, name: str) -> Path | None:
    """The file to serve for `name`, or None if the name isn't one we serve."""
    if not FILE_NAME.match(name):
        return None
    return thumbnails_dir(video_id) / name
