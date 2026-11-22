"""app/export_snapshot.py: what an export plays, from an edit list v5 (the segments
in order). How v4 reels became segments is tested in test_migrations.py."""

from typing import Any

import pytest

from app.export_snapshot import build_snapshot, subtract
from app.models import ExportSnapshot, ReelEdit


def seg(
    start: float,
    end: float,
    slow_mo: bool = False,
    rally_id: int | None = None,
    **kw: Any,
) -> dict[str, Any]:
    return {"id": kw.get("id", f"{start}-{end}-{slow_mo}"), "start": start, "end": end,
            "slow_mo": slow_mo, "rally_id": rally_id}  # fmt: skip


def edit_of(*segments: dict[str, Any], **fields: Any) -> ReelEdit:
    return ReelEdit.model_validate({"segments": list(segments), **fields})


def test_segments_in_list_order_slow_mo_doubled() -> None:
    edit = edit_of(seg(29, 41, rally_id=2), seg(9, 14, rally_id=1),
                   seg(14, 18, slow_mo=True, rally_id=1), seg(21, 29))  # fmt: skip
    snapshot = build_snapshot(edit, duration=60)
    assert [(s.start, s.end, s.slow_mo, s.rally_id) for s in snapshot.segments] == [
        (29, 41, False, 2),
        (9, 14, False, 1),
        (14, 18, True, 1),
        (21, 29, False, None),
    ]
    assert snapshot.duration == 12 + 5 + 8 + 8


def test_a_replay_overlaps_in_source_time() -> None:
    edit = edit_of(seg(9, 21, rally_id=1, id="a"), seg(15, 18, True, 1, id="b"))
    assert build_snapshot(edit, duration=60).duration == 12 + 6


def test_format_is_the_reels_unless_overridden() -> None:
    edit = edit_of(seg(1, 2), format="9:16")
    assert build_snapshot(edit, duration=10).format == "9:16"
    assert build_snapshot(edit, duration=10, format="16:9").format == "16:9"


def test_segments_are_cut_to_the_video() -> None:
    """A segment may end up to TRIM_TOLERANCE past the end (the app measured the
    duration a bit differently); the export stops at the real end."""
    snapshot = build_snapshot(edit_of(seg(24, 30.1)), duration=30.04)
    assert [(s.start, s.end) for s in snapshot.segments] == [(24, 30.04)]
    assert snapshot.duration == 6.04


def test_no_segments_no_snapshot_segments() -> None:
    """The API refuses such an export (409)."""
    snapshot = build_snapshot(ReelEdit(), duration=60)
    assert snapshot.segments == [] and snapshot.duration == 0


def test_captions_on_the_output_timeline() -> None:
    """Shown while the source time is inside them, in every segment that shows it
    (a replay shows it again), doubled in slow-mo, joined across touching segments;
    where two overlap the first in the list wins."""
    edit = edit_of(
        seg(9, 14, id="a"),
        seg(14, 18, slow_mo=True, id="b"),
        seg(29, 41, id="c"),
        seg(12, 14, id="replay"),
        captions=[
            {"text": "Huge smash", "start": 12, "end": 16},
            {"text": "First", "start": 30, "end": 32},
            {"text": "Second", "start": 31, "end": 33},
            {"text": "Never shown", "start": 45, "end": 47},
        ],
    )
    snapshot = build_snapshot(edit, duration=60)
    assert snapshot.model_dump()["captions"] == [
        {"text": "Huge smash", "start": 3, "end": 9},  # 12-14 normal, 14-16 slow
        {"text": "First", "start": 14, "end": 16},
        {"text": "Second", "start": 16, "end": 17},
        {"text": "Huge smash", "start": 25, "end": 27},  # the replay
    ]


def test_caption_shorter_than_a_frame_is_left_out() -> None:
    edit = edit_of(seg(1, 5), captions=[{"text": "x", "start": 4.995, "end": 6}])
    assert build_snapshot(edit, duration=10).captions == []


def test_snapshot_round_trips_as_stored_json() -> None:
    snapshot = build_snapshot(edit_of(seg(1, 2, slow_mo=True, rally_id=3)), duration=10)
    stored = snapshot.model_dump(mode="json")
    assert ExportSnapshot.model_validate(stored) == snapshot
    assert stored["duration"] == 2.0


def test_snapshots_stored_before_v5_still_read() -> None:
    """Exports made from v4 reels stored a `kind` per segment; it's ignored."""
    old = {"version": 1, "format": "16:9", "duration": 12.0, "captions": [],
           "segments": [{"kind": "gap", "start": 21, "end": 29, "slow_mo": False},
                        {"kind": "piece", "start": 29, "end": 33, "slow_mo": False}]}  # fmt: skip
    snapshot = ExportSnapshot.model_validate(old)
    assert [(s.start, s.rally_id) for s in snapshot.segments] == [
        (21, None),
        (29, None),
    ]


@pytest.mark.parametrize(
    ("span", "holes", "parts"),
    [
        ((0, 10), [], [(0, 10)]),
        ((0, 10), [(2, 3)], [(0, 2), (3, 10)]),
        ((0, 10), [(-5, 2), (8, 20)], [(2, 8)]),
        ((0, 10), [(0, 10)], []),
        ((0, 10), [(10, 12)], [(0, 10)]),  # touching isn't overlapping
    ],
)
def test_subtract(
    span: tuple[float, float],
    holes: list[tuple[float, float]],
    parts: list[tuple[float, float]],
) -> None:
    assert subtract(span, holes) == parts
