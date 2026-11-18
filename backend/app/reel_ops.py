"""
Keeping reels valid when rallies change (a user's correction, or re-analysis).

Reel clips point at rallies, and their pieces must stay inside the rally's bounds
(widened by the reel's lead_in/lead_out). When rallies change, every clip is
"retargeted": given the rally (or rallies) it now belongs to, its pieces are cut to
the new bounds. Pure functions on ReelEdit; crud.py does the database part.

    edit rally   -> same rally, new bounds: pieces outside are trimmed
    delete rally -> no rally: the clip is dropped
    split rally  -> two rallies: pieces before the split stay, the rest move to a new
                    clip for the second part, right after it (a piece across the
                    split point is cut in two)
    merge rallies-> the merged rally: a clip of the first followed directly by a
                    clip of the second become one
    re-analysis  -> the new rally overlapping the clip most (or none: dropped)

A clip's cuts (footage the user deleted) are retargeted exactly like its pieces:
trimmed to the new bounds, split with it, joined on a merge, dropped with the clip.

Restored breaks (ReelEdit.restored_anchors) need no retargeting: an anchor is one
time inside a stretch of footage, and the stretches are always worked out from the
pieces and cuts as they are now (stretches). So a restored break follows its
neighbours when they move. The frontend (reelEdit.ts, timeline.ts) applies the same
rule; shared/reel-segments/*.json checks both.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.models import Piece, ReelClip, ReelEdit, TimeRange

EPSILON = 1e-6  # shorter than this after trimming = nothing left
# Hidden stretches (and kept pieces) shorter than this count as nothing, like the
# app's timeline (its MIN_SPAN)
MIN_SPAN = 0.05

Span = tuple[float, float]


@dataclass(frozen=True)
class Target:
    """A rally a clip should belong to after the change."""

    rally_id: int
    start: float
    end: float


@dataclass
class Impact:
    clips_removed: int = 0
    pieces_trimmed: int = 0
    pieces_removed: int = 0


# For each clip: the rallies it belongs to now, in time order. [] = its rally is
# gone (drop the clip); None = not affected (leave the clip exactly as it is).
Retarget = Callable[[ReelClip], Sequence[Target] | None]


def _trim[R: TimeRange](
    pieces: list[R], lo: float, hi: float, impact: Impact
) -> list[R]:
    kept = []
    for piece in pieces:
        start, end = max(piece.start, lo), min(piece.end, hi)
        if end - start < EPSILON:
            impact.pieces_removed += 1
            continue
        if (start, end) != (piece.start, piece.end):
            impact.pieces_trimmed += 1
        kept.append(piece.model_copy(update={"start": start, "end": end}))
    return kept


def _split_at[R: TimeRange](pieces: list[R], at: float) -> tuple[list[R], list[R]]:
    """Pieces before and after `at`; a piece across it is cut in two (not counted as
    trimmed: nothing is lost)."""
    before, after = [], []
    for piece in pieces:
        if piece.end <= at:
            before.append(piece)
        elif piece.start >= at:
            after.append(piece)
        else:
            before.append(piece.model_copy(update={"end": at}))
            after.append(piece.model_copy(update={"start": at}))
    return before, after


def _join(first: list[Piece], second: list[Piece], impact: Impact) -> list[Piece]:
    """Pieces of two clips merged into one: sorted, overlaps cut off the later one."""
    joined: list[Piece] = []
    for piece in sorted(first + second, key=lambda p: p.start):
        if joined and piece.start < joined[-1].end:
            start = joined[-1].end
            if piece.end - start < EPSILON:
                impact.pieces_removed += 1
                continue
            impact.pieces_trimmed += 1
            piece = piece.model_copy(update={"start": start})
        joined.append(piece)
    return joined


def retarget_edit(
    edit: ReelEdit, retarget: Retarget, join: tuple[int, int] | None = None
) -> tuple[ReelEdit, Impact]:
    """The edit with every clip moved to its new rally(s), and what that cost.
    join: (first, second) rally ids of a merge; a clip of `first` directly followed
    by a clip of `second` becomes one clip. Other repeats of the same rally (e.g. a
    slow-mo replay right after) stay separate clips."""
    impact = Impact()
    clips: list[ReelClip] = []
    origins: list[int] = []  # each new clip's rally before the change
    for clip in edit.clips:
        targets = retarget(clip)
        if targets is None:
            clips.append(clip)
            origins.append(clip.rally_id)
            continue
        new_clips = []
        rest, rest_cuts = list(clip.pieces), list(clip.cuts)
        for k, target in enumerate(targets):
            if k + 1 < len(targets):  # hand the later pieces to the next target
                boundary = (target.end + targets[k + 1].start) / 2
                mine, rest = _split_at(rest, boundary)
                my_cuts, rest_cuts = _split_at(rest_cuts, boundary)
            else:
                mine, my_cuts = rest, rest_cuts
            lo, hi = target.start - edit.lead_in, target.end + edit.lead_out
            kept = _trim(mine, lo, hi, impact)
            if kept:
                cuts = _trim(my_cuts, lo, hi, Impact())  # not counted: nothing plays
                new_clips.append(
                    ReelClip(rally_id=target.rally_id, pieces=kept, cuts=cuts)
                )
        if not new_clips:
            impact.clips_removed += 1
        clips.extend(new_clips)
        origins.extend(clip.rally_id for _ in new_clips)

    if join:
        joined: list[ReelClip] = []
        for k, clip in enumerate(clips):
            if k and (origins[k - 1], origins[k]) == join:
                pieces = _join(joined[-1].pieces, clip.pieces, impact)
                cuts = _merge_ranges(joined[-1].cuts + clip.cuts)
                joined[-1] = ReelClip(rally_id=clip.rally_id, pieces=pieces, cuts=cuts)
            else:
                joined.append(clip)
        clips = joined
    return edit.model_copy(update={"clips": clips}), impact


def best_overlap(clip: ReelClip, rallies: Sequence[Target]) -> list[Target]:
    """Re-analysis: the rally overlapping the clip's pieces most, if any."""

    def overlap(r: Target) -> float:
        return sum(
            max(0.0, min(p.end, r.end) - max(p.start, r.start)) for p in clip.pieces
        )

    best = max(rallies, key=overlap, default=None)
    return [best] if best is not None and overlap(best) > 0 else []


# --- Restored breaks ------------------------------------------------------------------


def kept_pieces(edit: ReelEdit, duration: float | None) -> list[Piece]:
    """Every clip's pieces in reel order, cut to the video (pieces may end up to
    TRIM_TOLERANCE past it); slivers under MIN_SPAN left out."""
    limit = float("inf") if duration is None else duration
    kept = []
    for clip in edit.clips:
        for piece in clip.pieces:
            start, end = max(0.0, piece.start), min(piece.end, limit)
            if end - start >= MIN_SPAN:
                kept.append(piece.model_copy(update={"start": start, "end": end}))
    return kept


def _union(spans: Sequence[Span]) -> list[Span]:
    """Sorted, overlapping or touching spans merged."""
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _subtract(span: Span, holes: Sequence[Span]) -> list[Span]:
    parts = [span]
    for h_start, h_end in holes:
        parts = [
            part
            for start, end in parts
            for part in ((start, min(end, h_start)), (max(start, h_end), end))
            if part[1] - part[0] > 0
        ]
    return parts


def _gaps(covered: Sequence[Span], limit: float) -> list[Span]:
    """What `covered` (sorted, merged) leaves free in [0, limit], no slivers."""
    gaps, cursor = [], 0.0
    for start, end in covered:
        if start - cursor >= MIN_SPAN:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if limit - cursor >= MIN_SPAN:
        gaps.append((cursor, limit))
    return gaps


def _covered(edit: ReelEdit, duration: float | None) -> list[Span]:
    return _union([(p.start, p.end) for p in kept_pieces(edit, duration)])


def hidden_regions(edit: ReelEdit, duration: float | None) -> list[Span]:
    """The footage no piece covers, in time order: before the first piece, between
    neighbouring pieces (sorted by time; overlapping ones, e.g. a replay, count as
    one), and after the last one up to the end of the video. Under MIN_SPAN: none.
    Without a duration the last stretch runs to infinity. (Cuts are part of it.)"""
    limit = float("inf") if duration is None else duration
    return _gaps(_covered(edit, duration), limit)


def clip_cuts(edit: ReelEdit, duration: float | None) -> list[tuple[int, Span]]:
    """(clip index, span) of every cut that matters: cut to the video, minus what
    any piece covers (a replay's piece may show it), slivers dropped. In time order."""
    limit = float("inf") if duration is None else duration
    covered = _covered(edit, duration)
    found = []
    for index, clip in enumerate(edit.clips):
        for cut in clip.cuts:
            span = (max(0.0, cut.start), min(cut.end, limit))
            for part in _subtract(span, covered):
                if part[1] - part[0] >= MIN_SPAN:
                    found.append((index, part))
    return sorted(found, key=lambda c: c[1])


def stretches(edit: ReelEdit, duration: float | None) -> list[Span]:
    """What an anchor can restore: the hidden footage between the nearest piece or
    cut on each side (or the start / end of the video). Cuts are never part of one,
    so a restored break never crosses or includes a cut."""
    limit = float("inf") if duration is None else duration
    walls = _covered(edit, duration) + [span for _, span in clip_cuts(edit, duration)]
    return _gaps(_union(walls), limit)


def is_restored(region: Span, anchors: Sequence[float]) -> bool:
    """Strictly inside: an anchor on an edge belongs to no break."""
    return any(region[0] < a < region[1] for a in anchors)


def restored_regions(edit: ReelEdit, duration: float | None) -> list[Span]:
    return [
        stretch
        for stretch in stretches(edit, duration)
        if is_restored(stretch, edit.restored_anchors)
    ]


def live_anchors(edit: ReelEdit, duration: float | None) -> list[float]:
    """One anchor per restored stretch (the first in time order); anchors in no
    stretch (inside a piece or a cut, on an edge, past the end) are dropped."""
    kept = []
    for start, end in stretches(edit, duration):
        inside = sorted(a for a in edit.restored_anchors if start < a < end)
        if inside:
            kept.append(inside[0])
    return kept


def _merge_ranges(ranges: Sequence[TimeRange]) -> list[TimeRange]:
    return [
        TimeRange(start=start, end=end)
        for start, end in _union([(r.start, r.end) for r in ranges])
    ]


def normalize_edit(edit: ReelEdit, duration: float | None) -> ReelEdit:
    """What gets stored on save: clips without pieces dropped (their cuts with them),
    each clip's cuts sorted, merged and minus its own pieces (slivers dropped), and
    dead anchors dropped."""
    clips = []
    for clip in edit.without_empty_clips().clips:
        own = _union([(p.start, p.end) for p in clip.pieces])
        cuts = [
            TimeRange(start=start, end=end)
            for cut in _merge_ranges(clip.cuts)
            for start, end in _subtract((cut.start, cut.end), own)
            if end - start >= MIN_SPAN
        ]
        clips.append(clip.model_copy(update={"cuts": cuts}))
    edit = edit.model_copy(update={"clips": clips})
    return edit.model_copy(update={"restored_anchors": live_anchors(edit, duration)})
