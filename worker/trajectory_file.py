"""
Turn TrackNet's CSV (pixels) into the trajectory file the API serves (fractions).
Format: see backend/app/trajectory.py, which reads it (keep the two in sync).

Standard library only, so the backend's CI tests can check the format end to end.
"""

import csv
import gzip
import json
import os

FILENAME = "trajectory.json.gz"
VERSION = 1


def write_trajectory(csv_path, width, height, fps, out_path):
    """csv_path: TrackNet output (Frame, Visibility, X, Y in pixels of a
    width x height video). Writes out_path atomically (temp file + rename), so the
    API never reads a half-written file. Returns out_path."""
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    frames = 1 + max((int(r["Frame"]) for r in rows), default=-1)
    visible = [0] * frames
    xs = [None] * frames
    ys = [None] * frames
    for r in rows:
        i = int(r["Frame"])
        if int(r["Visibility"]):
            visible[i] = 1
            # 4 decimals: 0.1 px at 1280 wide, plenty for drawing a trail
            xs[i] = round(float(r["X"]) / width, 4)
            ys[i] = round(float(r["Y"]) / height, 4)
    tmp = out_path + ".tmp"
    with gzip.open(tmp, "wt") as fh:
        json.dump({"version": VERSION, "fps": fps, "visible": visible, "x": xs, "y": ys},
                  fh, separators=(",", ":"))
    os.replace(tmp, out_path)
    return out_path
