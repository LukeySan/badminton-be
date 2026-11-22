"""app/export_snapshot.py: what an export plays. The shared fixtures in
shared/reel-segments/ are the contract with the frontend's timeline (its tests
read the same files); the rest are backend-only details."""

import json
from pathlib import Path
from typing import Any

import pytest

from app.export_snapshot import build_snapshot, subtract
from app.models import ExportSnapshot, ReelEdit

FIXTURES = Path(__file__).resolve().parents[2] / "shared" / "reel-segments"


def fixtures() -> list[Path]:
    files = sorted(FIXTURES.glob("*.json"))
    assert files, f"no fixtures in {FIXTURES}"
    return files


@pytest.mark.parametrize("path", fixtures(), ids=lambda p: p.stem)
def test_shared_fixture(path: Path) -> None:
    case = json.loads(path.read_text(encoding="utf-8"))
    edit = ReelEdit.model_validate(case["edit"])
    # Every fixture must be a reel the API would accept for these rallies
    rallies = {r["id"]: (r["start"], r["end"]) for r in case["rallies"]}
    assert edit.problems(duration=case["duration"], rallies=rallies) == []

    snapshot = build_snapshot(edit, duration=case["duration"])
    expected = case["expected"]
    assert snapshot.model_dump()["segments"] == expected["segments"]
    assert snapshot.model_dump()["captions"] == expected["captions"]
    assert snapshot.duration == expected["duration"]


def edit_of(**fields: Any) -> ReelEdit:
    return ReelEdit.model_validate(fields)


def test_format_is_the_reels_unless_overridden() -> None:
    edit = edit_of(
        format="9:16", clips=[{"rally_id": 1, "pieces": [{"start": 1, "end": 2}]}]
    )
    assert build_snapshot(edit, duration=10).format == "9:16"
    assert build_snapshot(edit, duration=10, format="16:9").format == "16:9"


def test_pieces_are_clipped_to_the_video() -> None:
    """Pieces may end up to TRIM_TOLERANCE past the end (the app measured the
    duration a bit differently); the export stops at the real end."""
    edit = edit_of(
        clips=[{"rally_id": 1, "pieces": [{"start": 24, "end": 30.1}]}],
        restored_anchors=[12],
    )
    snapshot = build_snapshot(edit, duration=30.04)
    assert [(s.start, s.end) for s in snapshot.segments] == [(0, 24), (24, 30.04)]
    assert snapshot.duration == 30.04


def test_slivers_are_not_breaks() -> None:
    """Under MIN_SPAN (0.05 s), like the app's timeline: 0.03 s between two pieces
    is no hidden stretch, so an anchor there restores nothing."""
    edit = edit_of(
        clips=[
            {"rally_id": 1, "pieces": [{"start": 9, "end": 21}]},
            {"rally_id": 2, "pieces": [{"start": 21.03, "end": 30}]},
        ],
        restored_anchors=[21.01],
    )
    snapshot = build_snapshot(edit, duration=30)
    assert [s.kind for s in snapshot.segments] == ["piece", "piece"]


def test_no_pieces_means_only_a_break() -> None:
    """Without pieces the whole video is one hidden stretch; an anchor restores all
    of it. The API refuses such an export (409): restored breaks alone aren't a reel."""
    edit = edit_of(restored_anchors=[3])
    snapshot = build_snapshot(edit, duration=60)
    assert [(s.kind, s.start, s.end) for s in snapshot.segments] == [("gap", 0, 60)]


def test_without_a_duration_the_outro_isnt_rendered() -> None:
    """No video duration (never the case for analyzed videos): the stretch after the
    last piece has no end, so it can't be exported."""
    edit = edit_of(
        clips=[{"rally_id": 1, "pieces": [{"start": 9, "end": 21}]}],
        restored_anchors=[4, 30],
    )
    snapshot = build_snapshot(edit, duration=None)
    assert [(s.kind, s.start, s.end) for s in snapshot.segments] == [
        ("gap", 0, 9),
        ("piece", 9, 21),
    ]


def test_caption_shorter_than_a_frame_is_left_out() -> None:
    edit = edit_of(
        clips=[{"rally_id": 1, "pieces": [{"start": 1, "end": 5}]}],
        overlays={"captions": [{"text": "x", "start": 4.995, "end": 6}]},
    )
    assert build_snapshot(edit, duration=10).captions == []


def test_snapshot_round_trips_as_stored_json() -> None:
    edit = edit_of(
        clips=[{"rally_id": 1, "pieces": [{"start": 1, "end": 2, "slow_mo": True}]}]
    )
    snapshot = build_snapshot(edit, duration=10)
    stored = snapshot.model_dump(mode="json")
    assert ExportSnapshot.model_validate(stored) == snapshot
    assert stored["duration"] == 2.0


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
