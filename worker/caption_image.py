"""
Text overlay images for exported reels (worker/reel_export.py overlays them with
ffmpeg), styled like the app's captions: Manrope ExtraBold, light text on a dark
box with rounded corners (75% opaque).

Geometry (the app's preview must match): font size = overlay size x output height;
box padding PAD_X / PAD_Y ems, corner radius RADIUS ems, LINE_GAP ems between
lines; the text wraps to fit the frame minus the safe margins, where ICU's line
breaking allows (wrap_lines: Unicode line breaking, with dictionary word breaks for
Thai, which has no spaces, as browsers do for CSS); a word too wide for a line of
its own breaks between characters (CSS `overflow-wrap: anywhere`); the box is
centred on (x, y) and then moved inside the safe margin (SAFE_MARGIN of the shorter
side, on every edge). The line breaks are checked against the browser's
(Intl.Segmenter) with a fixture shared with the frontend:
backend/tests/fixtures/line_breaks/.

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

Line and character breaks come from ICU through PyICU, which is built from source
against the system's ICU (macOS: brew install icu4c pkgconf; Linux: apt install
libicu-dev pkg-config; see the README).
"""

import functools
import os
import struct
from collections.abc import Callable
from typing import NamedTuple

import icu  # PyICU
from PIL import Image, ImageDraw, ImageFont, features

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
# First font that has a character draws it
FONT_FILES = ["Manrope-ExtraBold.ttf", "NotoSansThai-ExtraBold.ttf"]

# The app (VideoScreen.tsx): text court-foreground, box court-ink/75, rounded 8px
TEXT_COLOR = (0xDE, 0xFF, 0xF2, 255)
BOX_COLOR = (0x19, 0x19, 0x23, round(0.75 * 255))

PAD_X, PAD_Y, RADIUS = 0.7, 0.3, 0.5  # in ems, like px-2.5 py-1 rounded-[8px]
LINE_GAP = 0.1  # ems between lines
# Overlays stay this far from every edge (fraction of the output's shorter side:
# 22 px at 720p, 32 px at 1080p, the same in 16:9 and 9:16)
SAFE_MARGIN = 0.03
MIN_FONT_PX = 8
SUPERSAMPLE = 4  # the box is drawn larger and scaled down: smooth corners

# Scripts that need shaping (raqm) to look right; text with any of these
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


def _split(text: str, breaks: icu.BreakIterator) -> list[str]:
    """Cut text at an ICU break iterator's boundaries. ICU counts in UTF-16 code
    units, so the cutting is done on its own string type (right for characters
    outside the BMP too)."""
    if not text:
        return []
    utf16 = icu.UnicodeString(text)
    breaks.setText(utf16)
    bounds = [0, *breaks]
    return [str(utf16[a:b]) for a, b in zip(bounds, bounds[1:], strict=False)]


def clusters(text: str) -> list[str]:
    """Split into user-perceived characters (grapheme clusters, ICU): a base and
    the marks that follow it (Thai vowels above/below, tone marks, SARA AM), so a
    line break or a font change never separates them."""
    return _split(text, icu.BreakIterator.createCharacterInstance(icu.Locale.getRoot()))


def line_pieces(text: str) -> list[str]:
    """The text cut where a line may break (ICU: Unicode line breaking, plus
    dictionary word breaks for Thai): each piece ends with the spaces and
    punctuation that may not start a line, e.g. "Rally ", "5 ", "สุด", "ยอด!"."""
    return _split(text, icu.BreakIterator.createLineInstance(icu.Locale.getRoot()))


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


def wrap_lines(text: str, width: Callable[[str], float], max_width: float) -> list[str]:
    """Greedy fill at the line break opportunities (line_pieces), like CSS with
    `overflow-wrap: anywhere`: trailing spaces don't count towards a line's width;
    a piece too wide for a line of its own starts a new line and breaks between
    clusters. `width` measures a string (pixels for drawing; the shared fixture's
    test uses characters)."""
    lines: list[str] = []
    current = ""
    for piece in line_pieces(text):
        if current and width((current + piece).rstrip()) <= max_width:
            current += piece
            continue
        if current:
            lines.append(current)
        current = ""
        if width(piece.rstrip()) <= max_width:
            current = piece
            continue
        for cluster in clusters(piece):
            if current.strip() and width((current + cluster).rstrip()) > max_width:
                lines.append(current)
                current = cluster
            else:
                current += cluster
    if current:
        lines.append(current)
    return [line.rstrip() for line in lines if line.strip()]


def wrap(text: str, size: int, max_width: float) -> list[str]:
    """Lines of `text` drawn at font size `size` (px), each at most max_width px."""
    return wrap_lines(text, lambda line: _width(font_runs(line), size), max_width)


def safe_margin(frame_w: int, frame_h: int) -> int:
    return round(min(frame_w, frame_h) * SAFE_MARGIN)


def place(centre: float, length: int, frame: int, margin: int) -> int:
    """Top-left coordinate along one axis: centred on `centre` (pixels), moved
    inside the margins; a box larger than the room between them is centred."""
    if length > frame - 2 * margin:
        return (frame - length) // 2
    return min(max(round(centre - length / 2), margin), frame - margin - length)


class OverlayImage(NamedTuple):
    image: Image.Image  # RGBA, the box with the text
    x: int  # where it goes in the frame (top-left)
    y: int


def render_text_overlay(
    text: str, frame_w: int, frame_h: int, x: float, y: float, size: float
) -> OverlayImage | None:
    """A text overlay's image and position in a frame_w x frame_h output: font size
    `size` x frame height, centred on (x, y) (fractions of the frame), kept inside
    the safe margin. None if nothing drawable is left (e.g. only emoji)."""
    text = drop_unsupported(text)
    if not text:
        return None
    if needs_complex_layout(text) and not check_text_layout():
        raise RuntimeError(
            "This text needs complex text layout (raqm) and Pillow doesn't have it: "
            "install FriBiDi (see worker/caption_image.py)"
        )
    font_px = max(MIN_FONT_PX, round(size * frame_h))
    margin = safe_margin(frame_w, frame_h)
    pad_x, pad_y = round(PAD_X * font_px), round(PAD_Y * font_px)
    lines = wrap(text, font_px, frame_w - 2 * margin - 2 * pad_x)
    line_runs = [font_runs(line) for line in lines]
    metrics = [
        [_font(r.font, font_px).getmetrics() for r in runs] for runs in line_runs
    ]
    ascents = [max(a for a, _ in m) for m in metrics]
    descents = [max(d for _, d in m) for m in metrics]
    gap = round(LINE_GAP * font_px)
    text_h = sum(ascents) + sum(descents) + gap * (len(lines) - 1)
    widths = [_width(runs, font_px) for runs in line_runs]
    box_w = round(max(widths)) + 2 * pad_x
    box_h = text_h + 2 * pad_y

    big = Image.new("RGBA", (box_w * SUPERSAMPLE, box_h * SUPERSAMPLE), (0, 0, 0, 0))
    ImageDraw.Draw(big).rounded_rectangle(
        (0, 0, big.width - 1, big.height - 1),
        radius=round(RADIUS * font_px * SUPERSAMPLE),
        fill=BOX_COLOR,
    )
    image = big.resize((box_w, box_h), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    top = pad_y
    for runs, width, ascent, descent in zip(
        line_runs, widths, ascents, descents, strict=True
    ):
        left = (box_w - width) / 2
        baseline = top + ascent
        for run in runs:
            font = _font(run.font, font_px)
            # Left-baseline anchor: Manrope and Noto Sans Thai share the baseline
            draw.text(
                (left, baseline), run.text, font=font, fill=TEXT_COLOR, anchor="ls"
            )
            left += font.getlength(run.text)
        top += ascent + descent + gap

    return OverlayImage(
        image,
        place(x * frame_w, box_w, frame_w, margin),
        place(y * frame_h, box_h, frame_h, margin),
    )
