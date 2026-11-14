"""
Thumbnail strip for the app's scrubber: one small frame per second, packed into
sprite sheets (JPEG grids) plus an index. Read by backend/app/thumbnails.py (keep the
two in sync).

    thumbnails/
      index.json      {"version": 1, "interval": 1.0, "count": 425,
                       "tile_width": 160, "tile_height": 90, "columns": 10, "rows": 10,
                       "sheets": ["sheet_000.jpg", ...]}
      sheet_000.jpg   tiles 0-99, left to right, top to bottom (1600x900)
      sheet_001.jpg   tiles 100-199 ...

Tile i shows the frame at i * interval seconds: sheet i // 100, column i % 10, row
(i // 10) % 10. Several 10x10 sheets rather than one tall image: a 60-minute video
would be one 1600x32400 image, more than phones decode reliably; this way the app
also loads only the sheets near the playhead.

Measured on match 4 (7.1 min, 720p): ~3 s, five sheets of ~320 KB (ffmpeg decodes
the whole video; the tiles themselves are cheap). Standard library + ffmpeg only, so
the backend's CI tests can run it.
"""

import json
import math
import os
import shutil
import subprocess

DIRNAME = "thumbnails"
INDEX = "index.json"
VERSION = 1
INTERVAL = 1.0  # seconds per tile
TILE_WIDTH = 160
COLUMNS, ROWS = 10, 10
JPEG_QUALITY = 10  # ffmpeg -q:v, 2 (best) .. 31; 10 ~ 3 KB a tile


def tile_count(duration):
    """How many tiles ffmpeg's fps filter makes: one per interval, rounded half up
    (checked against ffmpeg in the tests)."""
    return max(1, math.floor(duration / INTERVAL + 0.5))


def _probe(path, entries):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entries,
         "-of", "json", path],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return json.loads(result.stdout)


def video_duration(path):
    """The video stream's duration, or the file's: WebM/MKV files only have the
    latter."""
    info = _probe(path, "stream=duration:format=duration")
    duration = info["streams"][0].get("duration") or info["format"]["duration"]
    return float(duration)


def generate_thumbnails(video_path, out_dir):
    """Write the sheets + index.json to out_dir (replacing it). Built in a temporary
    folder next to it first, so out_dir never holds half a strip. Returns out_dir."""
    tmp = out_dir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    duration = video_duration(video_path)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video_path,
         "-vf", f"fps=1/{INTERVAL},scale={TILE_WIDTH}:-2,tile={COLUMNS}x{ROWS}",
         "-q:v", str(JPEG_QUALITY), "-start_number", "0",
         os.path.join(tmp, "sheet_%03d.jpg")],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    sheets = sorted(f for f in os.listdir(tmp) if f.endswith(".jpg"))
    size = _probe(os.path.join(tmp, sheets[0]), "stream=width,height")["streams"][0]
    index = {
        "version": VERSION,
        "interval": INTERVAL,
        "count": min(tile_count(duration), len(sheets) * COLUMNS * ROWS),
        "tile_width": size["width"] // COLUMNS,
        "tile_height": size["height"] // ROWS,
        "columns": COLUMNS,
        "rows": ROWS,
        "sheets": sheets,
    }
    with open(os.path.join(tmp, INDEX), "w") as fh:
        json.dump(index, fh)
    replace_dir(tmp, out_dir)
    return out_dir


def replace_dir(src, dst):
    """Move folder src to dst, replacing what's there. Two renames: the API sees the
    old strip or the new one (for an instant, none), never a mix of both."""
    old = dst + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(dst):
        os.replace(dst, old)
    os.replace(src, dst)
    shutil.rmtree(old, ignore_errors=True)
