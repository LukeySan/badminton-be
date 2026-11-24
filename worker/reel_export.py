"""
Render an export into an MP4 with ffmpeg, from the ORIGINAL upload (not the
512x288 analysis copy). What to render is the export's snapshot (built by the API,
backend/app/export_snapshot.py): segments of the original in output order, each
normal speed or slow-mo, and text overlays already on the output timeline.

One ffmpeg run:
- the original is opened once per segment with input seeking (-ss/-t before -i):
  frame-accurate (ffmpeg decodes from the keyframe before -ss and drops the frames
  before it, since we re-encode), and only the needed parts get decoded;
- each segment: timestamps from 0; slow-mo = setpts 2x + atempo 0.5; fitted to the
  output size (16:9: scaled, letterboxed if the original isn't 16:9, so the court is
  never cut; 9:16: centre crop, then scaled);
- concat filter joins them in order, video and audio together;
- text overlays (PNG images from caption_image.py, sized for this output and
  placed at their centre inside the safe margin) overlaid while they're on, in list
  order (later ones on top);
- H.264 (CRF 20, yuv420p, +faststart) and AAC.
No -r and no fps filter: frames keep their own timestamps (-fps_mode passthrough),
so variable-frame-rate phone video keeps its timing (forcing a rate on phone video
broke timestamps before). Progress comes from ffmpeg's -progress output.

Standard library + ffmpeg + Pillow (overlays), so the backend's tests can run it.
"""

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from typing import Any, TypedDict

import caption_image
from job_errors import SourceUnreadable

VIDEO_CODEC = ["-c:v", "libx264", "-preset", "fast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-profile:v", "high"]  # fmt: skip
AUDIO_CODEC = ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]
SLOW_MO = 2.0
STDERR_TAIL = 3000  # characters of ffmpeg's error output kept in the job's error


class RenderFailed(Exception):
    """ffmpeg exited with an error (details in the message)."""


class RenderResult(TypedDict):
    duration: float  # measured on the file
    size_bytes: int


def probe(path: str) -> dict[str, Any]:
    """{"duration", "has_audio"} of a media file; SourceUnreadable if it can't be
    read."""
    if not os.path.isfile(path):
        raise SourceUnreadable(f"{path} doesn't exist")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type:format=duration", "-of", "json", path],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    info = json.loads(result.stdout or "{}")
    types = [s.get("codec_type") for s in info.get("streams", [])]
    if result.returncode != 0 or "video" not in types:
        raise SourceUnreadable(f"ffprobe can't read {path}: {result.stderr.strip()}")
    return {
        "duration": float(info.get("format", {}).get("duration") or 0),
        "has_audio": "audio" in types,
    }


def fit_filter(fmt: str, width: int, height: int) -> str:
    if fmt == "9:16":
        # Centre crop to the output's shape (x/y default to centred), then scale
        crop_w = f"trunc(min(iw\\,ih*{width}/{height})/2)*2"
        crop_h = f"trunc(min(ih\\,iw*{height}/{width})/2)*2"
        return f"crop={crop_w}:{crop_h},scale={width}:{height}"
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease"
        f":force_divisible_by=2,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
    )


def build_command(
    source: str,
    snapshot: dict[str, Any],
    width: int,
    height: int,
    has_audio: bool,
    overlays: list[tuple[str, int, int, float, float]],
    out_path: str,
) -> list[str]:
    """overlays: (png path, x, y, start, end) on the output timeline, top-left
    corner in pixels; drawn in this order (later on top)."""
    segments = snapshot["segments"]
    fit = fit_filter(snapshot["format"], width, height)
    inputs: list[str] = []
    graph: list[str] = []
    for i, seg in enumerate(segments):
        start, end = float(seg["start"]), float(seg["end"])
        slow = bool(seg.get("slow_mo"))
        inputs += ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", source]
        video = f"[{i}:v]setpts=PTS-STARTPTS"
        if slow:
            video += f",setpts={SLOW_MO}*PTS"
        graph.append(f"{video},{fit},setsar=1,format=yuv420p[v{i}]")
        audio_fmt = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"
        if has_audio:
            audio = f"[{i}:a]asetpts=PTS-STARTPTS"
            if slow:
                audio += ",atempo=0.5"
        else:  # silent track, so every export has audio (players, social apps)
            length = (end - start) * (SLOW_MO if slow else 1.0)
            audio = f"anullsrc=r=48000:cl=stereo,atrim=duration={length:.3f}"
        graph.append(f"{audio},{audio_fmt}[a{i}]")
    pairs = "".join(f"[v{i}][a{i}]" for i in range(len(segments)))
    graph.append(f"{pairs}concat=n={len(segments)}:v=1:a=1[vcat][aout]")
    current = "vcat"
    for k, (png, x, y, start, end) in enumerate(overlays):
        inputs += ["-i", png]
        on = f"gte(t\\,{start:.3f})*lt(t\\,{end:.3f})"
        graph.append(
            f"[{current}][{len(segments) + k}:v]overlay=x={x}:y={y}"
            f":enable='{on}':eof_action=repeat[o{k}]"
        )
        current = f"o{k}"
    graph.append(f"[{current}]format=yuv420p[vout]")
    return [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-progress", "pipe:1", "-nostats",
        *inputs,
        "-filter_complex", ";".join(graph),
        "-map", "[vout]", "-map", "[aout]",
        *VIDEO_CODEC, *AUDIO_CODEC,
        "-fps_mode", "passthrough",
        "-movflags", "+faststart",
        "-f", "mp4", out_path,
    ]  # fmt: skip


def run_ffmpeg(
    command: list[str], expected: float, on_progress: Callable[[float], None]
) -> None:
    """Run ffmpeg, reporting progress (0-1). on_progress may raise (e.g. the job was
    taken away): ffmpeg is stopped and the exception passed on."""
    with tempfile.TemporaryFile(mode="w+") as stderr:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=stderr, text=True
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                key, _, value = line.strip().partition("=")
                if key == "out_time_us" and value.lstrip("-").isdigit():
                    done = int(value) / 1e6
                    on_progress(min(max(done / expected, 0.0), 1.0) if expected else 0)
            code = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        if code != 0:
            stderr.seek(0)
            tail = stderr.read()[-STDERR_TAIL:].strip()
            raise RenderFailed(f"ffmpeg exited with {code}: {tail}")


def overlay_images(
    snapshot: dict[str, Any], width: int, height: int, work: str
) -> list[tuple[str, int, int, float, float]]:
    """Each drawable overlay as a PNG in `work`: (path, x, y, start, end). Types
    this worker doesn't know (e.g. a future "score") are skipped, as is text with
    nothing drawable left."""
    out = []
    for k, overlay in enumerate(snapshot.get("overlays", [])):
        if overlay.get("type") != "text":
            continue
        image = caption_image.render_text_overlay(
            overlay["text"], width, height,
            float(overlay["x"]), float(overlay["y"]), float(overlay["size"]),
        )  # fmt: skip
        if image is None:
            continue
        png = os.path.join(work, f"overlay_{k:03d}.png")
        image.image.save(png)
        out.append(
            (png, image.x, image.y, float(overlay["start"]), float(overlay["end"]))
        )
    return out


def render_export(
    source: str,
    snapshot: dict[str, Any],
    width: int,
    height: int,
    out_path: str,
    on_progress: Callable[[float], None] = lambda _: None,
    log: Callable[[str], None] = print,
) -> RenderResult:
    """Render the snapshot to out_path (written as out_path + '.part' first, so a
    file at out_path is always complete)."""
    info = probe(source)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    work = tempfile.mkdtemp(prefix="export-", dir=os.path.dirname(out_path))
    partial = out_path + ".part"
    try:
        overlays = overlay_images(snapshot, width, height, work)
        command = build_command(
            source, snapshot, width, height, info["has_audio"], overlays, partial
        )
        log(
            f"rendering {len(snapshot['segments'])} segments, "
            f"{len(overlays)} overlays, {snapshot['duration']:.1f} s, {width}x{height}"
        )
        run_ffmpeg(command, float(snapshot["duration"]), on_progress)
        rendered = probe(partial)
        os.replace(partial, out_path)
        return {
            "duration": rendered["duration"],
            "size_bytes": os.path.getsize(out_path),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if os.path.exists(partial):
            os.remove(partial)
