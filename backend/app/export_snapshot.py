"""
What an export renders, resolved from the reel when the export is requested and
stored with it (Export.snapshot): later reel edits or rally fixes don't change an
export that is queued or running. Pure: no database, no files.

The reel's segments in order, each cut to the video (the app may have measured its
duration a little differently), slow-mo ones at half speed (twice as long). Captions
are mapped onto the output timeline: shown while the source time is inside
[start, end), in every segment that shows it (a replay shows it again), doubled
inside slow-mo segments; where captions overlap, the first in the list wins, like
the app.
"""

from collections.abc import Iterable

from app.models import (
    Caption,
    ExportCaption,
    ExportSegment,
    ExportSnapshot,
    ReelEdit,
    ReelFormat,
)

SLOW_MO_FACTOR = 2.0  # slow-mo = half speed = twice as long

Span = tuple[float, float]


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
    limit = float("inf") if duration is None else duration
    segments = []
    for segment in edit.segments:
        start, end = max(0.0, segment.start), min(segment.end, limit)
        if end > start:
            segments.append(
                ExportSegment(
                    start=_round(start),
                    end=_round(end),
                    slow_mo=segment.slow_mo,
                    rally_id=segment.rally_id,
                )
            )
    return segments


def map_captions(
    captions: list[Caption], segments: list[ExportSegment]
) -> list[ExportCaption]:
    out: list[ExportCaption] = []
    position = 0.0  # where the segment starts in the output
    for segment in segments:
        factor = SLOW_MO_FACTOR if segment.slow_mo else 1.0
        taken: list[Span] = []  # source time already captioned in this segment
        for caption in captions:
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
    """duration: the video's, to cut segments that end a little past it. format:
    None = the reel's."""
    segments = build_segments(edit, duration)
    return ExportSnapshot(
        format=format or edit.format,
        segments=segments,
        captions=map_captions(edit.captions, segments),
        duration=_round(sum(output_length(s) for s in segments)),
    )
