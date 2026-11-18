"""app/reel_ops.py: adjusting reel clips to changed rallies, and which hidden
footage restored anchors bring back (no database)."""

from collections.abc import Sequence
from typing import Any

import pytest

from app.models import ReelClip, ReelEdit
from app.reel_ops import (
    Target,
    best_overlap,
    clip_cuts,
    hidden_regions,
    live_anchors,
    normalize_edit,
    restored_regions,
    retarget_edit,
    stretches,
)

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


# --- restored breaks ---------------------------------------------------------------


def with_anchors(e: ReelEdit, *anchors: float) -> ReelEdit:
    return e.model_copy(update={"restored_anchors": list(anchors)})


def test_hidden_regions_are_what_no_piece_covers() -> None:
    e = edit((A, [(9, 14), (16, 21)]), (B, [(29, 41)]))
    assert hidden_regions(e, 60) == [(0, 9), (14, 16), (21, 29), (41, 60)]
    # Pieces touching the video's ends leave no stretch there
    e = edit((A, [(0, 21)]), (B, [(29, 60.05)]))
    assert hidden_regions(e, 60) == [(21, 29)]


def test_overlapping_pieces_count_as_one() -> None:
    """A replay repeats footage another piece already covers."""
    e = edit((A, [(9, 21)]), (B, [(29, 41)]), (A, [(15, 18)]), (A, [(20, 24)]))
    assert hidden_regions(e, 50) == [(0, 9), (24, 29), (41, 50)]


def test_slivers_are_no_stretch() -> None:
    e = edit((A, [(0.04, 21)]), (B, [(21.03, 59.98)]))
    assert hidden_regions(e, 60) == []


def test_no_duration_means_an_open_ended_last_stretch() -> None:
    e = edit((A, [(9, 21)]))
    assert hidden_regions(e, None) == [(0, 9), (21, float("inf"))]


def test_a_stretch_is_restored_by_an_anchor_strictly_inside() -> None:
    e = edit((A, [(9, 21)]), (B, [(29, 41)]))
    assert restored_regions(with_anchors(e, 25), 60) == [(21, 29)]
    # On a piece's edge, or inside a piece: nothing
    assert restored_regions(with_anchors(e, 21, 29, 15), 60) == []


def test_restored_stretch_follows_its_pieces() -> None:
    """The point of anchors: a rally fix or lead change moves the pieces, and the
    restored break between them moves too."""
    before = with_anchors(edit((A, [(9, 21)]), (B, [(29, 41)])), 25)
    after = with_anchors(edit((A, [(9.5, 20.5)]), (B, [(29.5, 40.5)]), lead=0.5), 25)
    assert restored_regions(before, 60) == [(21, 29)]
    assert restored_regions(after, 60) == [(20.5, 29.5)]


def test_live_anchors_one_per_break_dead_ones_dropped() -> None:
    e = edit((A, [(9, 21)]), (B, [(29, 41)]))
    anchors = with_anchors(e, 26.5, 15, 23.5, 70, 21, 3)
    assert live_anchors(anchors, 60) == [3, 23.5]


def test_normalize_drops_empty_clips_and_dead_anchors() -> None:
    e = ReelEdit.model_validate(
        {
            "clips": [
                {"rally_id": A, "pieces": [{"start": 9, "end": 21}]},
                {"rally_id": B, "pieces": []},
            ],
            "restored_anchors": [15, 25, 26],
        }
    )
    normal = normalize_edit(e, 60)
    assert [c.rally_id for c in normal.clips] == [A]
    assert normal.restored_anchors == [25]


# --- cuts: deleted footage ------------------------------------------------------------


def cut_edit(
    *clips: tuple[int, Sequence[tuple[float, float]], Sequence[tuple[float, float]]],
    anchors: Sequence[float] = (),
) -> ReelEdit:
    return ReelEdit.model_validate(
        {
            "clips": [
                {
                    "rally_id": r,
                    "pieces": [{"start": s, "end": e} for s, e in pieces],
                    "cuts": [{"start": s, "end": e} for s, e in cuts],
                }
                for r, pieces, cuts in clips
            ],
            "restored_anchors": list(anchors),
        }
    )


def cuts(e: ReelEdit) -> list[list[tuple[float, float]]]:
    return [[(c.start, c.end) for c in clip.cuts] for clip in e.clips]


def test_cuts_are_walls_between_stretches() -> None:
    """Rally 2's deleted lead 26-29: the hidden 21-29 is split into the stretch
    21-26 and the cut, so a restored break stops at the cut."""
    e = cut_edit((A, [(9, 21)], []), (B, [(29, 41)], [(26, 29)]), anchors=[25])
    assert hidden_regions(e, 60) == [(0, 9), (21, 29), (41, 60)]  # cuts are hidden
    assert stretches(e, 60) == [(0, 9), (21, 26), (41, 60)]
    assert restored_regions(e, 60) == [(21, 26)]


def test_an_anchor_in_a_cut_restores_nothing() -> None:
    e = cut_edit((A, [(9, 13), (16, 21)], [(13, 16)]), anchors=[14.5])
    assert restored_regions(e, 60) == []
    assert live_anchors(e, 60) == []


def test_a_cut_another_clip_shows_is_no_wall() -> None:
    """A replay's piece covers part of rally 1's deleted tail: that part plays (in
    the replay), the rest is still a cut."""
    e = cut_edit((A, [(9, 18)], [(18, 21)]), (A, [(17, 19)], []))
    assert clip_cuts(e, 60) == [(0, (19, 21))]


def test_normalize_tidies_cuts() -> None:
    """Sorted, merged, minus the clip's own pieces, slivers dropped; a clip without
    pieces goes with its cuts."""
    e = cut_edit(
        (A, [(9, 14)], [(18, 21), (12, 16), (16, 18), (20.98, 21.02)]),
        (B, [], [(29, 30)]),
    )
    assert cuts(normalize_edit(e, 60)) == [[(14, 21.02)]]


def test_rally_fix_trims_cuts_like_pieces() -> None:
    e = cut_edit((A, [(10, 18)], [(9, 10), (18, 21)]))
    new, impact = retarget_edit(e, only(A, [Target(A, 12, 19)]))  # 11-20 now
    assert cuts(new) == [[(18, 20)]]  # the lead cut 9-10 is outside 11-20 now
    assert spans(new) == [(A, [(11, 18)])]
    assert impact.pieces_trimmed == 1  # cuts aren't counted: nothing of them plays


def test_split_hands_cuts_to_the_part_they_are_in() -> None:
    e = cut_edit((A, [(9, 12), (14, 21)], [(12, 14)]))
    new, _ = retarget_edit(e, only(A, [Target(A, 10, 13), Target(C, 15, 20)]))
    assert spans(new) == [(A, [(9, 12)]), (C, [(14, 21)])]
    assert cuts(new) == [[(12, 14)], []]


def test_merge_joins_cuts_and_delete_drops_them() -> None:
    e = cut_edit((A, [(9, 18)], [(18, 21)]), (B, [(32, 41)], [(29, 32)]))
    merged, _ = retarget_edit(
        e,
        lambda clip: [Target(C, 10, 40)] if clip.rally_id in (A, B) else None,
        join=(A, B),
    )
    assert cuts(merged) == [[(18, 21), (29, 32)]]
    deleted, _ = retarget_edit(e, only(A, []))
    assert cuts(deleted) == [[(29, 32)]]
