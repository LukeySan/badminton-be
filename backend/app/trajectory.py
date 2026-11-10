"""
Per-frame shuttle positions, saved by the worker next to the video:

    storage/<video id>/trajectory.json.gz

Gzipped JSON, written by worker/trajectory_file.py (keep the two in sync):

    {"version": 1, "fps": 30.0,
     "visible": [0, 1, 1, ...],          one entry per frame; frame i is at i / fps s
     "x": [null, 0.5123, ...],           fractions of the frame width / height, like
     "y": [null, 0.3311, ...]}           court corners; null where it wasn't seen

A file rather than a row per frame: match 5 (21 minutes, 38,534 frames) is 50 KB
(0.5 MB unzipped; frames without a shuttle compress to almost nothing), where the
database would need 38,534 rows. The app only ever asks for a time range of one
video, which is a slice of these lists.
"""

import gzip
import json
import math
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import settings
from app.models import TrajectoryPoint

FILENAME = "trajectory.json.gz"


def trajectory_path(video_id: uuid.UUID) -> Path:
    return settings.STORAGE_DIR / str(video_id) / FILENAME


@dataclass(frozen=True)
class Trajectory:
    fps: float
    visible: list[int]
    x: list[float | None]
    y: list[float | None]


@lru_cache(maxsize=4)
def _read(path: str, _mtime_ns: int) -> Trajectory:
    """Cached per file version (the mtime changes when the worker replaces it), so
    scrubbing through one video doesn't unzip the file on every request."""
    with gzip.open(path, "rt") as fh:
        data = json.load(fh)
    if data.get("version") != 1:
        raise ValueError(f"{path}: unknown trajectory version {data.get('version')}")
    return Trajectory(
        fps=float(data["fps"]), visible=data["visible"], x=data["x"], y=data["y"]
    )


def load_trajectory(video_id: uuid.UUID) -> Trajectory | None:
    """None if the video has no trajectory (analyzed before trails existed, or
    not analyzed yet)."""
    path = trajectory_path(video_id)
    try:
        mtime = path.stat().st_mtime_ns
    except FileNotFoundError:
        return None
    return _read(str(path), mtime)


def points_between(
    trajectory: Trajectory, start: float, end: float
) -> list[TrajectoryPoint]:
    """Every frame whose time (frame / fps) is within [start, end]."""
    fps = trajectory.fps
    # Tiny epsilon: 2.0 s * 30 fps should be frame 60, not 59.999 -> 59
    first = max(0, math.ceil(start * fps - 1e-6))
    last = min(len(trajectory.visible) - 1, math.floor(end * fps + 1e-6))
    return [
        TrajectoryPoint(
            frame=i,
            t=round(i / fps, 3),
            visible=bool(trajectory.visible[i]),
            x=trajectory.x[i],
            y=trajectory.y[i],
        )
        for i in range(first, last + 1)
    ]
