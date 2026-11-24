"""app/export_snapshot.py: what an export plays, from an edit list v6 (the segments
in order, overlays on the reel's timeline). How older reels were converted is tested
in test_migrations.py."""

from typing import Any

import pytest

from app.export_snapshot import build_snapshot, to_output_time
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


def text(id: str, start: float, end: float, **kw: Any) -> dict[str, Any]:
    return {"id": id, "type": "text", "text": kw.pop("text", id), "start": start,
            "end": end, "x": kw.pop("x", 0.5), "y": kw.pop("y", 0.85), **kw}  # fmt: skip


def test_overlays_keep_reel_time_list_order_and_place() -> None:
    """Overlays are already on the reel's timeline (slow-mo counted double); the
    export keeps their order (drawing order: later on top) and position."""
    edit = edit_of(
        seg(9, 14, id="a"),  # reel 0-5
        seg(14, 18, slow_mo=True, id="b"),  # reel 5-13
        seg(29, 41, id="c"),  # reel 13-25
        overlays=[
            text("late", 20, 24, x=0.2, y=0.1, size=0.15),
            text("early", 3, 9),
            text("on-top", 3, 4),
        ],
    )
    snapshot = build_snapshot(edit, duration=60)
    assert snapshot.model_dump()["overlays"] == [
        {"type": "text", "text": "late", "start": 20, "end": 24, "x": 0.2, "y": 0.1,
         "size": 0.15},
        {"type": "text", "text": "early", "start": 3, "end": 9, "x": 0.5, "y": 0.85,
         "size": 0.06},
        {"type": "text", "text": "on-top", "start": 3, "end": 4, "x": 0.5,
         "y": 0.85, "size": 0.06},
    ]  # fmt: skip


def test_overlays_follow_a_segment_cut_at_the_videos_end() -> None:
    """A segment ending past the video (within the 0.1 s tolerance) is cut in the
    export: overlays after it move up by the cut, one inside the cut-off part ends
    with the segment, and one only inside the cut-off part isn't drawn."""
    edit = edit_of(
        seg(25, 30.1, id="end"),  # reel 0-5.1, output 0-5.04
        seg(10, 12, id="next"),  # reel 5.1-7.1, output 5.04-7.04
        overlays=[
            text("after", 5.5, 6.5),
            text("across", 4.5, 5.6),
            text("cut-off", 4.8, 5.1),
        ],
    )
    edit.overlays[2].start = 5.06  # under 0.3 s: not valid in an edit, set by hand
    snapshot = build_snapshot(edit, duration=30.04)
    assert [(o.text, o.start, o.end) for o in snapshot.overlays] == [
        ("after", 5.44, 6.44),
        ("across", 4.5, 5.54),
    ]
    assert snapshot.duration == 7.04


def test_overlay_past_the_output_is_cut_to_it() -> None:
    """The edit keeps overlays inside the reel; the output can still end a little
    earlier (cut segment)."""
    edit = edit_of(seg(25, 30.1), overlays=[text("o", 4, 5.1)])
    assert [
        (o.start, o.end) for o in build_snapshot(edit, duration=30.04).overlays
    ] == [(4, 5.04)]


def test_snapshot_round_trips_as_stored_json() -> None:
    snapshot = build_snapshot(edit_of(seg(1, 2, slow_mo=True, rally_id=3)), duration=10)
    stored = snapshot.model_dump(mode="json")
    assert ExportSnapshot.model_validate(stored) == snapshot
    assert stored["duration"] == 2.0


def test_snapshots_stored_before_v5_still_read() -> None:
    """Exports made from v4 reels stored a `kind` per segment; it's ignored. (Their
    captions became overlays in migration 4c1e7a9b2d60: snapshot v2.)"""
    old = {"version": 2, "format": "16:9", "duration": 12.0, "overlays": [],
           "segments": [{"kind": "gap", "start": 21, "end": 29, "slow_mo": False},
                        {"kind": "piece", "start": 29, "end": 33, "slow_mo": False}]}  # fmt: skip
    snapshot = ExportSnapshot.model_validate(old)
    assert [(s.start, s.rally_id) for s in snapshot.segments] == [
        (21, None),
        (29, None),
    ]


def test_reel_time_to_output_time() -> None:
    """Segments of the reel: (reel start, reel length, output length)."""
    timeline = [(0.0, 5.1, 5.04), (5.1, 2.0, 2.0), (7.1, 1.0, 0.0)]
    assert to_output_time(0, timeline) == 0
    assert to_output_time(5.0, timeline) == 5.0
    assert to_output_time(5.08, timeline) == 5.04  # in the cut-off part: its end
    assert to_output_time(5.1, timeline) == 5.04
    assert to_output_time(6.1, timeline) == pytest.approx(6.04)
    assert to_output_time(7.5, timeline) == 7.04  # a segment wholly past the video
    assert to_output_time(99, timeline) == 7.04
