"""
Caption images for exported reels (worker/reel_export.py overlays them with ffmpeg),
styled like the app's caption: Manrope ExtraBold, light text on a dark box with
rounded corners (75% opaque), centred near the bottom.

Pillow, not ffmpeg's drawtext: Homebrew's ffmpeg is built without libfreetype, and
Pillow draws real rounded corners. Fonts are bundled (worker/fonts/, OFL):
Manrope, plus Noto Sans Thai for the characters Manrope doesn't have. Each run of
text is drawn in the first font that covers it (read from the fonts' cmap tables).

Thai needs complex text layout (raqm: HarfBuzz shaping stacks vowels and tone marks
correctly). Pillow's wheels include raqm but load FriBiDi at runtime:
  macOS:  brew install fribidi, and run with
          DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:/usr/local/lib:/usr/lib
  Linux:  apt install libfribidi0 (ffmpeg's packages usually bring it)
check_text_layout() reports whether it's available; the worker won't start without.
"""

import functools
import os
import struct
import unicodedata
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont, features

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
# First font that has a character draws it
FONT_FILES = ["Manrope-ExtraBold.ttf", "NotoSansThai-ExtraBold.ttf"]

# The app (VideoScreen.tsx): text court-foreground, box court-ink/75, rounded 8px
TEXT_COLOR = (0xDE, 0xFF, 0xF2, 255)
BOX_COLOR = (0x19, 0x19, 0x23, round(0.75 * 255))

# Sizes relative to the output's shorter side, so 720p and 1080p look the same
FONT_SIZE = 0.044  # 32 px at 720p (the app: 14 px on a ~320-px-tall player)
PAD_X, PAD_Y, RADIUS = 0.7, 0.3, 0.5  # in ems, like px-2.5 py-1 rounded-[8px]
LINE_GAP = 0.1  # ems between lines
MAX_WIDTH = 0.84  # of the frame width: inside the title-safe area
# Bottom margin, as a share of the height: 9:16 leaves room for the buttons and
# text that Reels / TikTok / Shorts draw over the bottom of the video
BOTTOM_MARGIN = {"16:9": 0.09, "9:16": 0.17}
MIN_SCALE = 0.6  # shrink long captions down to 60% before breaking words
SUPERSAMPLE = 4  # the box is drawn larger and scaled down: smooth corners

# Scripts that need shaping (raqm) to look right; a caption with any of these
# characters is refused without it rather than rendered wrong
COMPLEX_RANGES = [(0x0E00, 0x0E7F)]  # Thai


def check_text_layout() -> bool:
    """True if Pillow has raqm (complex text layout)."""
    return bool(features.check("raqm"))


def needs_complex_layout(text: str) -> bool:
    return any(lo <= ord(ch) <= hi for ch in text for lo, hi in COMPLEX_RANGES)


@functools.cache
def font_codepoints(path: str) -> frozenset[int]:
    """The characters a TrueType font has glyphs for, from its cmap table
    (formats 4 and 12, which every modern font has)."""
    with open(path, "rb") as fh:
        data = fh.read()
    (num_tables,) = struct.unpack_from(">H", data, 4)
    cmap = None
    for i in range(num_tables):
        tag, _, offset, _ = struct.unpack_from(">4sIII", data, 12 + 16 * i)
        if tag == b"cmap":
            cmap = offset
    if cmap is None:
        raise ValueError(f"{path}: no cmap table")
    (n_subtables,) = struct.unpack_from(">H", data, cmap + 2)
    subtables = {}
    for i in range(n_subtables):
        platform, encoding, offset = struct.unpack_from(">HHI", data, cmap + 4 + 8 * i)
        subtables[(platform, encoding)] = cmap + offset
    codes: set[int] = set()
    # Prefer the full-Unicode table (format 12), else the BMP one (format 4)
    for key in [(3, 10), (0, 4), (0, 6), (3, 1), (0, 3)]:
        start = subtables.get(key)
        if start is None:
            continue
        (fmt,) = struct.unpack_from(">H", data, start)
        if fmt == 12:
            (n_groups,) = struct.unpack_from(">I", data, start + 12)
            for g in range(n_groups):
                first, last, _ = struct.unpack_from(">III", data, start + 16 + 12 * g)
                codes.update(range(first, last + 1))
            return frozenset(codes)
        if fmt == 4:
            seg_count = struct.unpack_from(">H", data, start + 6)[0] // 2
            ends = start + 14
            starts = ends + 2 * seg_count + 2
            deltas = starts + 2 * seg_count
            range_offsets = deltas + 2 * seg_count
            for s in range(seg_count):
                (end,) = struct.unpack_from(">H", data, ends + 2 * s)
                (first,) = struct.unpack_from(">H", data, starts + 2 * s)
                (delta,) = struct.unpack_from(">h", data, deltas + 2 * s)
                ro_at = range_offsets + 2 * s
                (range_offset,) = struct.unpack_from(">H", data, ro_at)
                for c in range(first, end + 1):
                    if c == 0xFFFF:
                        continue
                    if range_offset == 0:
                        glyph = (c + delta) & 0xFFFF
                    else:
                        at = ro_at + range_offset + 2 * (c - first)
                        (glyph,) = struct.unpack_from(">H", data, at)
                        glyph = (glyph + delta) & 0xFFFF if glyph else 0
                    if glyph:
                        codes.add(c)
            return frozenset(codes)
    raise ValueError(f"{path}: no Unicode cmap (format 4 or 12)")


@functools.cache
def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    engine = ImageFont.Layout.RAQM if check_text_layout() else ImageFont.Layout.BASIC
    return ImageFont.truetype(os.path.join(FONTS_DIR, name), size, layout_engine=engine)


def clusters(text: str) -> list[str]:
    """Split into user-perceived characters: a base and the marks that follow it
    (Thai vowels above/below and tone marks are combining marks), so a line break
    or a font change never separates them."""
    out: list[str] = []
    for ch in text:
        joins = unicodedata.category(ch).startswith("M") or ch in "‍️"
        if out and joins:
            out[-1] += ch
        else:
            out.append(ch)
    return out


class Run(NamedTuple):
    text: str
    font: str  # file name in FONTS_DIR


def _covers() -> dict[str, frozenset[int]]:
    return {n: font_codepoints(os.path.join(FONTS_DIR, n)) for n in FONT_FILES}


def drop_unsupported(text: str) -> str:
    """Remove characters no bundled font has (e.g. emoji): they would be drawn as
    empty boxes. Spaces left doubled by that are collapsed."""
    covered = frozenset().union(*_covers().values())
    kept = "".join(c for c in clusters(text) if ord(c[0]) in covered or c[0].isspace())
    return " ".join(kept.split())


def font_runs(text: str) -> list[Run]:
    """Split text into runs drawn with one font each. A cluster stays in the
    current run's font if that font has it (spaces, punctuation), so shaping isn't
    broken up needlessly; otherwise the first font that has all of it."""
    covers = _covers()

    def has(name: str, cluster: str) -> bool:
        return all(ord(ch) in covers[name] for ch in cluster)

    runs: list[Run] = []
    for cluster in clusters(text):
        current = runs[-1].font if runs else None
        if current is not None and has(current, cluster):
            font = current
        else:
            font = next((n for n in FONT_FILES if has(n, cluster)), FONT_FILES[0])
        if runs and runs[-1].font == font:
            runs[-1] = Run(runs[-1].text + cluster, font)
        else:
            runs.append(Run(cluster, font))
    return runs


def _width(runs: list[Run], size: int) -> float:
    return sum(_font(r.font, size).getlength(r.text) for r in runs)


def _wrap(text: str, size: int, max_width: float) -> list[str] | None:
    """Greedy word wrap; None if a single word doesn't fit."""
    lines: list[str] = []
    for word in text.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and _width(font_runs(candidate), size) <= max_width:
            lines[-1] = candidate
        elif _width(font_runs(word), size) <= max_width:
            lines.append(word)
        else:
            return None
    return lines


def _break_clusters(text: str, size: int, max_width: float) -> list[str]:
    """Last resort (e.g. a long Thai phrase without spaces): break anywhere between
    clusters."""
    lines = [""]
    for cluster in clusters(text):
        if lines[-1] and _width(font_runs(lines[-1] + cluster), size) > max_width:
            lines.append(cluster.lstrip())
        else:
            lines[-1] += cluster
    return [line.strip() for line in lines if line.strip()]


def layout(text: str, frame_w: int, frame_h: int) -> tuple[int, list[str]]:
    """Font size and lines: wrap at spaces into at most 2 lines, shrinking the text
    (down to MIN_SCALE) until it fits; then break between characters."""
    base = round(min(frame_w, frame_h) * FONT_SIZE)
    for scale in (1.0, 0.9, 0.8, 0.7, MIN_SCALE):
        size = max(8, round(base * scale))
        max_width = frame_w * MAX_WIDTH - 2 * PAD_X * size
        lines = _wrap(text, size, max_width)
        if lines is not None and len(lines) <= 2:
            return size, lines
    size = max(8, round(base * MIN_SCALE))
    return size, _break_clusters(text, size, frame_w * MAX_WIDTH - 2 * PAD_X * size)


class CaptionImage(NamedTuple):
    image: Image.Image  # RGBA, the box with the text
    x: int  # where it goes in the frame (top-left)
    y: int


def render_caption(
    text: str, frame_w: int, frame_h: int, fmt: str
) -> CaptionImage | None:
    """The caption's image and position; None if nothing drawable is left (e.g. a
    caption that is only emoji)."""
    text = drop_unsupported(text)
    if not text:
        return None
    if needs_complex_layout(text) and not check_text_layout():
        raise RuntimeError(
            "This caption needs complex text layout (raqm) and Pillow doesn't have "
            "it: install FriBiDi (see worker/caption_image.py)"
        )
    size, lines = layout(text, frame_w, frame_h)
    line_runs = [font_runs(line) for line in lines]
    metrics = [[_font(r.font, size).getmetrics() for r in runs] for runs in line_runs]
    ascents = [max(a for a, _ in m) for m in metrics]
    descents = [max(d for _, d in m) for m in metrics]
    gap = round(LINE_GAP * size)
    text_h = sum(ascents) + sum(descents) + gap * (len(lines) - 1)
    widths = [_width(runs, size) for runs in line_runs]
    pad_x, pad_y = round(PAD_X * size), round(PAD_Y * size)
    box_w = round(max(widths)) + 2 * pad_x
    box_h = text_h + 2 * pad_y

    big = Image.new("RGBA", (box_w * SUPERSAMPLE, box_h * SUPERSAMPLE), (0, 0, 0, 0))
    ImageDraw.Draw(big).rounded_rectangle(
        (0, 0, big.width - 1, big.height - 1),
        radius=round(RADIUS * size * SUPERSAMPLE),
        fill=BOX_COLOR,
    )
    image = big.resize((box_w, box_h), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    y = pad_y
    for runs, width, ascent, descent in zip(
        line_runs, widths, ascents, descents, strict=True
    ):
        x = (box_w - width) / 2
        baseline = y + ascent
        for run in runs:
            font = _font(run.font, size)
            # Left-baseline anchor: Manrope and Noto Sans Thai share the baseline
            draw.text((x, baseline), run.text, font=font, fill=TEXT_COLOR, anchor="ls")
            x += font.getlength(run.text)
        y += ascent + descent + gap

    margin = round(frame_h * BOTTOM_MARGIN.get(fmt, BOTTOM_MARGIN["16:9"]))
    return CaptionImage(image, (frame_w - box_w) // 2, frame_h - margin - box_h)
