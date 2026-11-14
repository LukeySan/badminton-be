"""app/reel_ops.py: adjusting reel clips to changed rallies (no database)."""

from collections.abc import Sequence
from typing import Any

import pytest

from app.models import ReelClip, ReelEdit
from app.reel_ops import Target, best_overlap, retarget_edit

A, B, C = 1, 2, 3  # rally ids


def edit(
    *clips: tuple[int, Sequence[tuple[float, float]]], lead: float = 1.0
) -> ReelEdit:
    return ReelEdit.model_validate(
        {
            "lead_in": lead,
            "lead_out": lead,
            "clips": [
                {"rally_id": r, "pieces": [{"start": s, "end": e} for s, e in pieces]}
                for r, pieces in clips
            ],
        }
    )


def spans(e: ReelEdit) -> list[tuple[int, list[tuple[float, float]]]]:
    return [(c.rally_id, [(p.start, p.end) for p in c.pieces]) for c in e.clips]


def only(rally_id: int, targets: list[Target]) -> Any:
    return lambda clip: targets if clip.rally_id == rally_id else None


def test_unaffected_clips_stay_exactly_as_they_are() -> None:
    before = edit((A, [(10, 12)]), (B, [(30, 31)]))
    after, impact = retarget_edit(before, only(C, []))
    assert after == before
    assert impact.clips_removed == impact.pieces_trimmed == impact.pieces_removed == 0


def test_edit_trims_pieces_to_the_new_bounds_plus_lead() -> None:
    """Rally A was 10-20, now 12-18: with 1 s lead, pieces may span 11-19."""
    before = edit((A, [(9.5, 10.5), (11, 14), (15, 19.5), (19.5, 21)]), (B, [(30, 31)]))
    after, impact = retarget_edit(before, only(A, [Target(A, 12, 18)]))
    assert spans(after) == [(A, [(11, 14), (15, 19)]), (B, [(30, 31)])]
    assert (impact.pieces_trimmed, impact.pieces_removed) == (1, 2)
    assert impact.clips_removed == 0


def test_slow_mo_survives_trimming() -> None:
    before = ReelEdit.model_validate(
        {
            "clips": [
                {"rally_id": A, "pieces": [{"start": 9, "end": 15, "slow_mo": True}]}
            ]
        }
    )
    after, _ = retarget_edit(before, only(A, [Target(A, 12, 18)]))
    assert after.clips[0].pieces[0].slow_mo is True
    assert after.clips[0].pieces[0].start == 11


def test_clip_with_nothing_left_is_dropped() -> None:
    before = edit((A, [(9.5, 10.5)]), (B, [(30, 31)]))
    after, impact = retarget_edit(before, only(A, [Target(A, 15, 18)]))
    assert spans(after) == [(B, [(30, 31)])]
    assert (impact.clips_removed, impact.pieces_removed) == (1, 1)


def test_deleted_rally_drops_its_clips() -> None:
    before = edit((A, [(10, 12)]), (B, [(30, 31)]), (A, [(15, 16)]))
    after, impact = retarget_edit(before, only(A, []))
    assert spans(after) == [(B, [(30, 31)])]
    assert impact.clips_removed == 2
    assert impact.pieces_removed == 0  # counted as clips, not pieces


def test_split_moves_later_pieces_to_a_new_clip_right_after() -> None:
    """Rally A (10-20) split at 15 into A (10-15) and C (15-20). A piece across
    the split point is cut in two; nothing is lost."""
    before = edit((A, [(10, 12), (13, 17), (18, 20)]), (B, [(30, 31)]))
    after, impact = retarget_edit(
        before, only(A, [Target(A, 10, 15), Target(C, 15, 20)])
    )
    assert spans(after) == [
        (A, [(10, 12), (13, 15)]),
        (C, [(15, 17), (18, 20)]),
        (B, [(30, 31)]),
    ]
    assert impact.pieces_trimmed == impact.pieces_removed == impact.clips_removed == 0


def test_split_with_all_pieces_on_one_side_keeps_one_clip() -> None:
    before = edit((A, [(16, 19)]))
    after, _ = retarget_edit(before, only(A, [Target(A, 10, 15), Target(C, 15, 20)]))
    assert spans(after) == [(C, [(16, 19)])]


def test_merge_joins_back_to_back_clips() -> None:
    """A (10-20) and B (22-30) merged into C (10-30). Back-to-back clips of A and B
    become one; overlapping lead-out/lead-in pieces are cut where they meet."""
    before = edit((A, [(10, 15), (18, 21)]), (B, [(20.5, 25)]), (A, [(11, 12)]))
    merged = Target(C, 10, 30)

    def retarget(clip: ReelClip) -> list[Target] | None:
        return [merged] if clip.rally_id in (A, B) else None

    after, impact = retarget_edit(before, retarget, join=(A, B))
    # The replay of A after them stays its own clip
    assert spans(after) == [(C, [(10, 15), (18, 21), (21, 25)]), (C, [(11, 12)])]
    assert impact.pieces_trimmed == 1


def test_merge_join_drops_a_piece_swallowed_by_the_one_before() -> None:
    before = edit((A, [(10, 22)]), (B, [(20.5, 21.5)]))
    after, impact = retarget_edit(before, lambda clip: [Target(C, 10, 30)], join=(A, B))
    assert spans(after) == [(C, [(10, 22)])]
    assert impact.pieces_removed == 1


def test_merge_does_not_join_clips_in_the_other_order() -> None:
    """B's clip before A's: playing them as one clip would change the reel order."""
    before = edit((B, [(22, 25)]), (A, [(10, 15)]))
    after, _ = retarget_edit(before, lambda clip: [Target(C, 10, 30)], join=(A, B))
    assert spans(after) == [(C, [(22, 25)]), (C, [(10, 15)])]


def test_same_rally_twice_is_not_joined_without_a_merge() -> None:
    """A user may use one rally twice in a row on purpose (e.g. a slow-mo replay)."""
    before = edit((A, [(10, 15)]), (A, [(10, 15)]))
    after, _ = retarget_edit(before, only(A, [Target(A, 10, 20)]))
    assert len(after.clips) == 2


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        ([(9, 21)], A),  # covers A
        ([(19, 31)], A),  # a tie (1 s of each): the earlier rally
        ([(18, 31)], A),  # 2 s of A, 1 s of B
        ([(19, 33)], B),  # 1 s of A, 3 s of B
        ([(21, 29)], None),  # in the break between them
    ],
)
def test_best_overlap(pieces: list[tuple[float, float]], expected: int | None) -> None:
    rallies = [Target(A, 10, 20), Target(B, 30, 45)]
    clip = edit((99, pieces)).clips[0]
    result = best_overlap(clip, rallies)
    assert [t.rally_id for t in result] == ([expected] if expected else [])
