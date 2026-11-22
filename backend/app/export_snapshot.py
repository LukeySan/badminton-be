"""
What an export renders, resolved from the reel when the export is requested and
stored with it (Export.snapshot): later reel edits or rally fixes don't change an
export that is queued or running. Pure: no database, no files.

Always the app's "Rallies only" result, even if the editor is in "Full video" view:

- pieces: the clips' pieces in reel order. Their times already include lead_in /
  lead_out (the editor grows and shrinks pieces with the lead), so they are used as
  stored. Slow-mo pieces play at half speed.
- gaps: restored breaks at normal speed. The footage between neighbouring pieces
  (sorted by time; also before the first and after the last) is hidden unless one of
  the reel's restored_anchors is inside it; then all of it plays
  (reel_ops.restored_regions, the same rule as the app's timeline). Each break is
  placed right before the first piece (in reel order) that starts where it ends or
  later; for a reel in time order that is simply time order.
- captions: shown while the source time is inside [start, end), as in the app's
  preview, mapped onto the output timeline (doubled inside slow-mo pieces). If
  captions overlap, the first in the list wins, like the app.

shared/reel-segments/*.json (repo root) pins this down with examples, and the
frontend's timeline tests check its preview against the same files.
"""

from collections.abc import Iterable

from app.models import (
    ExportCaption,
    ExportSegment,
    ExportSnapshot,
    ReelEdit,
    ReelFormat,
)
from app.reel_ops import Span, kept_pieces, restored_regions

SLOW_MO_FACTOR = 2.0  # slow-mo = half speed = twice as long


def _round(t: float) -> float:
    return round(t, 3)


def subtract(span: Span, holes: Iterable[Span]) -> list[Span]:
    """The parts of `span` outside all `holes`."""
    parts = [span]
    for h_start, h_end in holes:
        next_parts = []
        for start, end in parts:
            if h_end <= start or end <= h_start:
                next_parts.append((start, end))
                continue
            if start < h_start:
                next_parts.append((start, h_start))
            if h_end < end:
                next_parts.append((h_end, end))
        parts = next_parts
    return parts


def output_length(segment: ExportSegment) -> float:
    factor = SLOW_MO_FACTOR if segment.slow_mo else 1.0
    return (segment.end - segment.start) * factor


def build_segments(edit: ReelEdit, duration: float | None) -> list[ExportSegment]:
    pieces = [
        ExportSegment(
            kind="piece",
            start=_round(p.start),
            end=_round(p.end),
            slow_mo=p.slow_mo,
        )
        for p in kept_pieces(edit, duration)
    ]
    gaps = [
        ExportSegment(kind="gap", start=_round(start), end=_round(end))
        for start, end in restored_regions(edit, duration)
        if end != float("inf")  # no duration: the last stretch has no end
    ]  # in time order

    segments: list[ExportSegment] = []
    for segment in pieces:
        while gaps and gaps[0].end <= segment.start:
            segments.append(gaps.pop(0))
        segments.append(segment)
    return segments + gaps


def map_captions(edit: ReelEdit, segments: list[ExportSegment]) -> list[ExportCaption]:
    out: list[ExportCaption] = []
    position = 0.0  # where the segment starts in the output
    for segment in segments:
        factor = SLOW_MO_FACTOR if segment.slow_mo else 1.0
        taken: list[Span] = []  # source time already captioned in this segment
        for caption in edit.overlays.captions:
            span = (max(caption.start, segment.start), min(caption.end, segment.end))
            if span[1] <= span[0]:
                continue
            for start, end in subtract(span, taken):
                out_start = _round(position + (start - segment.start) * factor)
                out_end = _round(position + (end - segment.start) * factor)
                if out_end - out_start < 0.01:  # under a frame
                    continue
                last = out[-1] if out else None
                if last and last.text == caption.text and last.end == out_start:
                    last.end = out_end  # continues across a segment boundary
                else:
                    out.append(
                        ExportCaption(text=caption.text, start=out_start, end=out_end)
                    )
            taken.append(span)
        position += output_length(segment)
    return sorted(out, key=lambda c: c.start)


def build_snapshot(
    edit: ReelEdit, *, duration: float | None, format: ReelFormat | None = None
) -> ExportSnapshot:
    """duration: the video's, to clip pieces that end a little past it (the app may
    have measured it slightly differently). format: None = the reel's."""
    segments = build_segments(edit, duration)
    return ExportSnapshot(
        format=format or edit.format,
        segments=segments,
        captions=map_captions(edit, segments),
        duration=_round(sum(output_length(s) for s in segments)),
    )
