"""worker/reel_export.py and worker/caption_image.py with real ffmpeg on small
generated videos: durations (slow-mo counts double), sizes per format and quality,
audio, variable frame rate, text overlays (position and size checked on grabbed
frames, Thai)."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "worker"))
import caption_image  # noqa: E402
import reel_export  # noqa: E402
from job_errors import SourceUnreadable  # noqa: E402

# Thai shaping needs raqm (FriBiDi). CI has it; locally scripts/test.sh sets it up
# on macOS. Run without it, these tests are skipped locally but fail in CI.
needs_raqm = pytest.mark.skipif(
    not caption_image.check_text_layout() and not os.environ.get("CI"),
    reason="Pillow has no raqm here (see worker/caption_image.py)",
)

# The VFR source: 2 s at 30 fps, then 2 s at 15 fps, then 2 s at 60 fps, each with
# its own tone
PARTS = [(30, 440), (15, 660), (60, 880)]


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def make_part(path: Path, fps: int, tone: int | None, size: str) -> None:
    audio = ["-f", "lavfi", "-i", f"sine=f={tone}:d=2"] if tone else []
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}:duration=2",
           *audio, "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           *(["-c:a", "aac"] if tone else []), str(path))  # fmt: skip


@pytest.fixture(scope="module")
def vfr_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Variable frame rate with audio, 640x360 (16:9), 6 s."""
    folder = tmp_path_factory.mktemp("vfr")
    parts = []
    for i, (fps, tone) in enumerate(PARTS):
        parts.append(folder / f"part{i}.mp4")
        make_part(parts[-1], fps, tone, "640x360")
    listing = folder / "list.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    out = folder / "vfr.mp4"
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out))
    return out


@pytest.fixture(scope="module")
def silent_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """No audio stream, 4:3 (480x360)."""
    out = tmp_path_factory.mktemp("silent") / "silent.mp4"
    make_part(out, 30, None, "480x360")
    return out


def probe(path: Path) -> dict[str, Any]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,codec_name,width,height,pix_fmt,duration,nb_read_frames"
         ":format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    info = json.loads(out)
    streams = {s["codec_type"]: s for s in info["streams"]}
    return {
        "duration": float(info["format"]["duration"]),
        "video": streams.get("video"),
        "audio": streams.get("audio"),
    }


def seg(
    start: float, end: float, slow: bool = False, kind: str = "piece"
) -> dict[str, Any]:
    return {"kind": kind, "start": start, "end": end, "slow_mo": slow}


def snapshot(
    segments: list[dict[str, Any]],
    fmt: str = "16:9",
    overlays: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    duration = sum(
        (s["end"] - s["start"]) * (2 if s["slow_mo"] else 1) for s in segments
    )
    return {"version": 2, "format": fmt, "segments": segments,
            "overlays": overlays or [], "duration": round(duration, 3)}  # fmt: skip


def text(
    value: str,
    start: float,
    end: float,
    x: float = 0.5,
    y: float = 0.85,
    size: float = 0.06,
) -> dict[str, Any]:
    return {"type": "text", "text": value, "start": start, "end": end, "x": x,
            "y": y, "size": size}  # fmt: skip


def render(
    source: Path, snap: dict[str, Any], out: Path, size: tuple[int, int] = (1280, 720)
) -> reel_export.RenderResult:
    return reel_export.render_export(
        str(source), snap, size[0], size[1], str(out), log=lambda _: None
    )


def tolerance(snap: dict[str, Any]) -> float:
    """Each segment may end up to one source frame long (15 fps parts: 1/15 s)."""
    return len(snap["segments"]) / 15 + 0.02


@pytest.mark.parametrize(
    ("fmt", "size"),
    [("16:9", (1280, 720)), ("9:16", (720, 1280)), ("16:9", (1920, 1080)),
     ("9:16", (1080, 1920))],
)  # fmt: skip
def test_duration_size_and_audio(
    vfr_source: Path, tmp_path: Path, fmt: str, size: tuple[int, int]
) -> None:
    snap = snapshot(
        [seg(0.5, 1.5), seg(2.2, 2.7, slow=True), seg(3.0, 3.5, kind="gap"),
         seg(4.2, 5.0, slow=True)],
        fmt,
    )  # fmt: skip
    assert snap["duration"] == 1 + 1 + 0.5 + 1.6
    out = tmp_path / "e.mp4"
    result = render(vfr_source, snap, out, size)

    info = probe(out)
    video, audio = info["video"], info["audio"]
    assert (video["codec_name"], video["pix_fmt"]) == ("h264", "yuv420p")
    assert (video["width"], video["height"]) == size
    assert audio is not None and audio["codec_name"] == "aac"
    assert abs(info["duration"] - snap["duration"]) <= tolerance(snap)
    assert abs(float(audio["duration"]) - float(video["duration"])) < 0.1
    assert result == {"duration": info["duration"], "size_bytes": out.stat().st_size}
    assert not Path(str(out) + ".part").exists()
    assert os.listdir(tmp_path) == ["e.mp4"]  # no overlay or work files left over


def test_variable_frame_rate_is_kept(vfr_source: Path, tmp_path: Path) -> None:
    """No frame rate is forced: a second from the 15 fps part has 15 frames, one
    from the 60 fps part 60 (a forced 30 fps would give 30 and 30), and slow-mo
    stretches frames instead of adding any."""
    for start, end, slow, frames in [
        (2.2, 3.2, False, 15),
        (4.2, 5.2, False, 60),
        (4.2, 4.7, True, 30),
    ]:
        out = tmp_path / f"{start}-{slow}.mp4"
        snap = snapshot([seg(start, end, slow)])
        render(vfr_source, snap, out)
        info = probe(out)
        assert abs(int(info["video"]["nb_read_frames"]) - frames) <= 1, (start, slow)
        assert abs(info["duration"] - snap["duration"]) <= tolerance(snap)


def test_source_without_audio_gets_a_silent_track(
    silent_source: Path, tmp_path: Path
) -> None:
    out = tmp_path / "e.mp4"
    snap = snapshot([seg(0.2, 1.0), seg(1.0, 1.5, slow=True)])
    render(silent_source, snap, out)
    info = probe(out)
    assert info["audio"] is not None
    assert abs(float(info["audio"]["duration"]) - snap["duration"]) < 0.1


def test_letterbox_for_16_9_and_crop_for_9_16(
    silent_source: Path, tmp_path: Path
) -> None:
    """A 4:3 original: 16:9 keeps the whole picture (black bars left and right),
    9:16 crops the middle."""

    def left_column_brightness(path: Path, width: int, height: int) -> float:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, check=True,
        ).stdout  # fmt: skip
        column = [raw[y * width + 10] for y in range(height)]
        return sum(column) / height

    wide = tmp_path / "wide.mp4"
    render(silent_source, snapshot([seg(0, 1)]), wide)
    assert left_column_brightness(wide, 1280, 720) < 20  # black bar
    tall = tmp_path / "tall.mp4"
    render(silent_source, snapshot([seg(0, 1)], "9:16"), tall, (720, 1280))
    assert left_column_brightness(tall, 720, 1280) > 20  # picture to the edge


def frame_region(
    path: Path, t: float, box: tuple[int, int, int, int], width: int
) -> list[int]:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t}", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    x, y, w, h = box
    return [
        raw[row * width + col] for row in range(y, y + h) for col in range(x, x + w)
    ]


@needs_raqm
def test_overlays_are_burned_in_while_on(vfr_source: Path, tmp_path: Path) -> None:
    value = "ลูกนี้สุดยอด! Rally 5"
    segments = [seg(0.0, 2.0)]
    plain, with_text = tmp_path / "plain.mp4", tmp_path / "text.mp4"
    render(vfr_source, snapshot(segments), plain)
    render(vfr_source, snapshot(segments, overlays=[text(value, 0.2, 1.0)]), with_text)

    image = caption_image.render_text_overlay(value, 1280, 720, 0.5, 0.85, 0.06)
    assert image is not None
    box = (image.x, image.y, image.image.width, image.image.height)

    def difference(t: float) -> float:
        a = frame_region(plain, t, box, 1280)
        b = frame_region(with_text, t, box, 1280)
        return sum(abs(p - q) for p, q in zip(a, b, strict=True)) / len(a)

    assert difference(0.6) > 20  # the box is there
    assert difference(1.5) < 3  # and gone after its end


def test_overlay_only_emoji_or_unknown_type_is_skipped(
    vfr_source: Path, tmp_path: Path
) -> None:
    out = tmp_path / "e.mp4"
    score = {**text("", 0, 1), "type": "score"}  # reserved: not drawn (yet)
    snap = snapshot([seg(0, 1)], overlays=[text("🏸🏸", 0, 1), score])
    assert reel_export.overlay_images(snap, 1280, 720, str(tmp_path)) == []
    render(vfr_source, snap, out)
    assert probe(out)["video"] is not None


def test_overlays_are_drawn_in_list_order() -> None:
    """Each overlay goes on top of the result so far: later ones on top."""
    overlays = [("a.png", 10, 20, 0.0, 1.0), ("b.png", 30, 40, 0.5, 1.0)]
    command = reel_export.build_command(
        "in.mp4", snapshot([seg(0, 1)]), 1280, 720, True, overlays, "out.mp4"
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "[vcat][1:v]overlay=x=10:y=20" in graph
    assert "[o0][2:v]overlay=x=30:y=40" in graph
    assert command.index("a.png") < command.index("b.png")


# --- positions and sizes on real frames -----------------------------------------------


@pytest.fixture(scope="module")
def white_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Plain white, 16:9 (640x360), 2 s: the dark overlay box stands out."""
    out = tmp_path_factory.mktemp("white") / "white.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=white:size=640x360:rate=30:duration=2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out))  # fmt: skip
    return out


def dark_box(
    path: Path, t: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the dark pixels in the frame at t: the overlay's
    box on a white picture (right/bottom exclusive)."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t}", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    dark = bytes(1 if v < 160 else 0 for v in range(256))
    rows = []
    left, right = width, 0
    for row in range(height):
        line = raw[row * width : (row + 1) * width].translate(dark)
        first = line.find(1)
        if first >= 0:
            rows.append(row)
            left, right = min(left, first), max(right, line.rfind(1) + 1)
    assert rows, "no overlay box in the frame"
    return left, rows[0], right, rows[-1] + 1


def one_line_box_height(size: float, frame_h: int) -> int:
    """A one-line Latin box: font size = size x frame height, plus the padding."""
    font_px = round(size * frame_h)
    ascent, descent = ImageFont.truetype(
        os.path.join(caption_image.FONTS_DIR, "Manrope-ExtraBold.ttf"), font_px
    ).getmetrics()
    return ascent + descent + 2 * round(caption_image.PAD_Y * font_px)


@pytest.mark.parametrize(
    ("fmt", "size"),
    [("16:9", (1280, 720)), ("16:9", (1920, 1080)), ("9:16", (720, 1280)),
     ("9:16", (1080, 1920))],
)  # fmt: skip
def test_overlay_position_and_size_on_the_frame(
    white_source: Path, tmp_path: Path, fmt: str, size: tuple[int, int]
) -> None:
    """Centred on (x, y) as fractions of the output frame, the text as tall as
    `size` of the frame height, in both formats and qualities; one placed past the
    corner stays inside the safe margin."""
    w, h = size
    out = tmp_path / "e.mp4"
    snap = snapshot(
        [seg(0, 1)],
        fmt,
        overlays=[
            text("Nice", 0.0, 0.5, x=0.3, y=0.25, size=0.06),
            text("Win", 0.5, 1.0, x=1.0, y=1.0, size=0.12),
        ],
    )
    render(white_source, snap, out, size)

    left, top, right, bottom = dark_box(out, 0.25, w, h)
    assert abs((left + right) / 2 - 0.3 * w) <= 2
    assert abs((top + bottom) / 2 - 0.25 * h) <= 2
    assert abs((bottom - top) - one_line_box_height(0.06, h)) <= 2

    left, top, right, bottom = dark_box(out, 0.75, w, h)
    margin = caption_image.safe_margin(w, h)
    assert margin == round(min(w, h) * 0.03)
    assert abs(right - (w - margin)) <= 2  # pushed in from the corner
    assert abs(bottom - (h - margin)) <= 2
    assert abs((bottom - top) - one_line_box_height(0.12, h)) <= 2


def test_failing_ffmpeg_raises_with_its_message(
    vfr_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reel_export, "VIDEO_CODEC", ["-c:v", "no_such_encoder"])
    out = tmp_path / "e.mp4"
    with pytest.raises(reel_export.RenderFailed, match="no_such_encoder"):
        render(vfr_source, snapshot([seg(0, 1)]), out)
    assert os.listdir(tmp_path) == []  # nothing half-written left


def test_missing_or_unreadable_source(tmp_path: Path) -> None:
    with pytest.raises(SourceUnreadable, match="doesn't exist"):
        render(tmp_path / "gone.mp4", snapshot([seg(0, 1)]), tmp_path / "e.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(SourceUnreadable, match="ffprobe can't read"):
        render(junk, snapshot([seg(0, 1)]), tmp_path / "e.mp4")


def test_progress_is_reported_and_can_stop_the_render(
    vfr_source: Path, tmp_path: Path
) -> None:
    seen: list[float] = []

    def stop(fraction: float) -> None:
        seen.append(fraction)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        reel_export.render_export(str(vfr_source), snapshot([seg(0, 6)]), 1280, 720,
                                  str(tmp_path / "e.mp4"), on_progress=stop,
                                  log=lambda _: None)  # fmt: skip
    assert len(seen) == 1 and 0 <= seen[0] <= 1
    assert os.listdir(tmp_path) == []


# --- caption images ------------------------------------------------------------------


def test_fonts_cover_latin_and_thai() -> None:
    manrope, thai = (
        caption_image.font_codepoints(os.path.join(caption_image.FONTS_DIR, name))
        for name in caption_image.FONT_FILES
    )
    assert ord("A") in manrope and ord("ก") not in manrope
    assert {ord("ก"), ord("่"), ord("ิ")} <= thai  # consonant, tone, vowel


def test_text_is_split_into_font_runs() -> None:
    runs = caption_image.font_runs("Rally 5 สุดยอด! ที่นี่")
    assert runs == [
        caption_image.Run("Rally 5 ", "Manrope-ExtraBold.ttf"),
        # Punctuation and spaces stay in the Thai run: Noto Sans Thai has them
        caption_image.Run("สุดยอด! ที่นี่", "NotoSansThai-ExtraBold.ttf"),
    ]


def test_marks_stay_with_their_letter() -> None:
    assert caption_image.clusters("ที่นี่") == ["ที่", "นี่"]


def test_unsupported_characters_are_dropped() -> None:
    assert caption_image.drop_unsupported("Smash 🏸 ok") == "Smash ok"
    assert caption_image.render_text_overlay("🏸", 1280, 720, 0.5, 0.5, 0.06) is None


@needs_raqm
def test_thai_is_shaped() -> None:
    """With raqm, HarfBuzz stacks the tone mark above the vowel in ที่; without it,
    they would collide (the mark would sit lower)."""
    path = os.path.join(caption_image.FONTS_DIR, "NotoSansThai-ExtraBold.ttf")
    basic = ImageFont.truetype(path, 64, layout_engine=ImageFont.Layout.BASIC)
    raqm = ImageFont.truetype(path, 64, layout_engine=ImageFont.Layout.RAQM)
    assert raqm.getbbox("ที่", anchor="ls")[1] < basic.getbbox("ที่", anchor="ls")[1] - 5


def test_text_overlay_layout() -> None:
    """Font size = size x frame height (the same share of the frame at 720p and
    1080p); centred on (x, y); rounded transparent corners."""
    small = caption_image.render_text_overlay("Nice", 1280, 720, 0.5, 0.5, 0.06)
    large = caption_image.render_text_overlay("Nice", 1920, 1080, 0.5, 0.5, 0.06)
    assert small is not None and large is not None
    assert abs(large.image.height / small.image.height - 1.5) < 0.05
    assert small.x == round(640 - small.image.width / 2)
    assert small.y == round(360 - small.image.height / 2)
    assert small.image.getpixel((0, 0))[3] == 0  # type: ignore[index]
    bigger = caption_image.render_text_overlay("Nice", 1280, 720, 0.5, 0.5, 0.12)
    assert bigger is not None and bigger.image.height > small.image.height * 1.9


def test_long_text_wraps_inside_the_safe_margin() -> None:
    long_text = "A very long caption that needs to wrap onto several lines nicely"
    wrapped = caption_image.render_text_overlay(long_text, 720, 1280, 0.5, 0.5, 0.06)
    one_line = caption_image.render_text_overlay("A", 720, 1280, 0.5, 0.5, 0.06)
    assert wrapped is not None and one_line is not None
    margin = caption_image.safe_margin(720, 1280)
    assert wrapped.image.width <= 720 - 2 * margin
    assert margin <= wrapped.x and wrapped.x + wrapped.image.width <= 720 - margin
    assert wrapped.image.height > one_line.image.height * 2  # 3+ lines


@pytest.mark.parametrize(
    ("centre", "length", "frame", "margin", "left"),
    [
        (500, 100, 1000, 20, 450),  # centred
        (10, 100, 1000, 20, 20),  # pushed in at the start
        (1000, 100, 1000, 20, 880),  # and at the end
        (500, 980, 1000, 20, 10),  # bigger than the room inside the margins: centred
    ],
)
def test_place(centre: float, length: int, frame: int, margin: int, left: int) -> None:
    assert caption_image.place(centre, length, frame, margin) == left


@needs_raqm
def test_long_thai_without_spaces_breaks_at_words() -> None:
    """80 characters, no spaces, at the overlay's size in pixels: lines break
    between Thai words (ICU's dictionary, like the browser), never inside one.
    The shared fixture checks the exact breaks (test_line_breaks.py)."""
    value = "ลูกนี้สุดยอดมาก" * 5
    lines = caption_image.wrap(value, 77, 600)
    assert len(lines) > 1
    assert "".join(lines) == value  # nothing lost
    words = {"ลูก", "นี้", "สุด", "ยอด", "มาก"}
    for line in lines:
        assert set(caption_image.line_pieces(line)) <= words
        assert caption_image._width(caption_image.font_runs(line), 77) <= 600


def test_a_long_word_starts_its_own_line() -> None:
    """Like CSS overflow-wrap: anywhere, a word too wide for any line starts on a
    new line, breaks between characters, and the next word may join its last
    piece (at 40 px, 10 W's fill a 400 px line)."""
    lines = caption_image.wrap("Hi " + "W" * 35 + " ok", 40, 400)
    assert lines == ["Hi", "W" * 10, "W" * 10, "W" * 10, "WWWWW ok"]
