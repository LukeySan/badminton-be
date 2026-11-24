"""
What an export renders, resolved from the reel when the export is requested and
stored with it (Export.snapshot): later reel edits or rally fixes don't change an
export that is queued or running. Pure: no database, no files.

The reel's segments in order, each cut to the video (the app may have measured its
duration a little differently), slow-mo ones at half speed (twice as long).
Overlays are already on the reel's timeline; they keep their list order (drawing
order) and are mapped onto the output's timeline, which differs from the reel's only
where a segment was cut at the video's end (then everything after it moves up by the
cut, at most TRIM_TOLERANCE).
"""

from app.models import (
    SLOW_MO_FACTOR,
    ExportSegment,
    ExportSnapshot,
    ExportTextOverlay,
    Overlay,
    ReelEdit,
    ReelFormat,
)

MIN_OUTPUT_OVERLAY = 0.01  # seconds: shorter than a frame isn't drawn


def _round(t: float) -> float:
    return round(t, 3)


def output_length(segment: ExportSegment) -> float:
    factor = SLOW_MO_FACTOR if segment.slow_mo else 1.0
    return (segment.end - segment.start) * factor


def build_segments(
    edit: ReelEdit, duration: float | None
) -> tuple[list[ExportSegment], list[tuple[float, float, float]]]:
    """The output's segments, and for each segment of the reel (reel start, reel
    length, output length) to map reel time to output time."""
    limit = float("inf") if duration is None else duration
    segments: list[ExportSegment] = []
    timeline: list[tuple[float, float, float]] = []
    reel_position = 0.0
    for segment in edit.segments:
        start, end = max(0.0, segment.start), min(segment.end, limit)
        out_length = 0.0
        if end > start:
            exported = ExportSegment(
                start=_round(start),
                end=_round(end),
                slow_mo=segment.slow_mo,
                rally_id=segment.rally_id,
            )
            segments.append(exported)
            out_length = output_length(exported)
        timeline.append((reel_position, segment.reel_length, out_length))
        reel_position += segment.reel_length
    return segments, timeline


def to_output_time(t: float, timeline: list[tuple[float, float, float]]) -> float:
    """Reel time -> output time (inside a cut-short segment, its last frame)."""
    out_position = 0.0
    for reel_start, reel_length, out_length in timeline:
        if t < reel_start + reel_length:
            return out_position + min(max(t - reel_start, 0.0), out_length)
        out_position += out_length
    return out_position


def map_overlays(
    overlays: list[Overlay], timeline: list[tuple[float, float, float]]
) -> list[ExportTextOverlay]:
    out: list[ExportTextOverlay] = []
    for overlay in overlays:
        start = _round(to_output_time(overlay.start, timeline))
        end = _round(to_output_time(overlay.end, timeline))
        if end - start < MIN_OUTPUT_OVERLAY:
            continue
        out.append(
            ExportTextOverlay(
                text=overlay.text,
                start=start,
                end=end,
                x=overlay.x,
                y=overlay.y,
                size=overlay.size,
            )
        )
    return out


def build_snapshot(
    edit: ReelEdit, *, duration: float | None, format: ReelFormat | None = None
) -> ExportSnapshot:
    """duration: the video's, to cut segments that end a little past it. format:
    None = the reel's (overlay positions are fractions, so they work in either)."""
    segments, timeline = build_segments(edit, duration)
    return ExportSnapshot(
        format=format or edit.format,
        segments=segments,
        overlays=map_overlays(edit.overlays, timeline),
        duration=_round(sum(output_length(s) for s in segments)),
    )
