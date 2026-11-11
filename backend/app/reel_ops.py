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
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.models import Piece, ReelClip, ReelEdit

EPSILON = 1e-6  # shorter than this after trimming = nothing left


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


def _trim(pieces: list[Piece], lo: float, hi: float, impact: Impact) -> list[Piece]:
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


def _split_at(pieces: list[Piece], at: float) -> tuple[list[Piece], list[Piece]]:
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
        rest = list(clip.pieces)
        for k, target in enumerate(targets):
            if k + 1 < len(targets):  # hand the later pieces to the next target
                boundary = (target.end + targets[k + 1].start) / 2
                mine, rest = _split_at(rest, boundary)
            else:
                mine = rest
            kept = _trim(
                mine,
                target.start - edit.lead_in,
                target.end + edit.lead_out,
                impact,
            )
            if kept:
                new_clips.append(ReelClip(rally_id=target.rally_id, pieces=kept))
        if not new_clips:
            impact.clips_removed += 1
        clips.extend(new_clips)
        origins.extend(clip.rally_id for _ in new_clips)

    if join:
        joined: list[ReelClip] = []
        for k, clip in enumerate(clips):
            if k and (origins[k - 1], origins[k]) == join:
                pieces = _join(joined[-1].pieces, clip.pieces, impact)
                joined[-1] = ReelClip(rally_id=clip.rally_id, pieces=pieces)
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
