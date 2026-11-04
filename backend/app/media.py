"""
Video helpers built on ffprobe/ffmpeg (install with: brew install ffmpeg).

ffmpeg applies a phone video's rotation metadata, and so does the worker's 720p
downscale, so the frame the user taps the court on has the same orientation as
the frames the tracker sees.
"""

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, TypedDict


class VideoInfo(TypedDict):
    duration: float
    fps: float
    width: int  # as displayed (rotation applied)
    height: int


def _tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} not found; install it with: brew install ffmpeg")
    return found


def _rotation(stream: dict[str, Any]) -> int:
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            return int(side["rotation"])
    return int(stream.get("tags", {}).get("rotate", 0))


def probe(path: Path) -> VideoInfo:
    """Duration, fps and displayed size. Raises ValueError if it isn't a video."""
    result = subprocess.run(
        [
            _tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate:stream_side_data=rotation"
            ":stream_tags=rotate:format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    if result.returncode != 0 or not streams:
        raise ValueError("not a readable video file")
    s = streams[0]
    rate = s.get("avg_frame_rate", "0/0")
    if rate in ("0/0", "0"):
        rate = s.get("r_frame_rate", "0/0")
    fps = float(Fraction(rate)) if rate not in ("0/0", "0") else 0.0
    duration = float(data.get("format", {}).get("duration") or 0)
    if fps <= 0 or duration <= 0:
        raise ValueError("couldn't read the video's frame rate or duration")
    w, h = int(s["width"]), int(s["height"])
    if abs(_rotation(s)) % 180 == 90:
        w, h = h, w
    return {"duration": duration, "fps": fps, "width": w, "height": h}


def frame_jpeg(path: Path, t: float) -> bytes:
    """One frame at t seconds as JPEG bytes."""
    result = subprocess.run(
        [
            _tool("ffmpeg"), "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
            "-frames:v", "1", "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "-",
        ],
        capture_output=True, timeout=30,
    )  # fmt: skip
    if result.returncode != 0 or not result.stdout:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(stderr or "ffmpeg returned no frame")
    return result.stdout
