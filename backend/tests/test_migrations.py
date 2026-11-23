"""Data conversions inside migrations (pure functions, loaded from the file)."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from app.export_snapshot import build_snapshot
from app.models import ReelEdit

VERSIONS = Path(__file__).resolve().parents[1] / "app" / "alembic" / "versions"


def load(revision: str) -> ModuleType:
    (path,) = VERSIONS.glob(f"{revision}_*.py")
    spec = importlib.util.spec_from_file_location(f"migration_{revision}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reel_v2 = load("6acbb05a9b0b")
one_reel = load("9bd7ddc4eed1")
reel_v3 = load("d769f6b0ca89")
reel_v4 = load("ec4778601b52")
reel_v5 = load("8f3a5c2d9e14")


def to_current(v2_edit: dict[str, Any], duration: float = 120.0) -> dict[str, Any]:
    """A v2 edit through the later migrations, to validate it as today's ReelEdit."""
    v4 = reel_v4.edit_v3_to_v4(reel_v3.edit_v2_to_v3(v2_edit), RALLIES, duration)
    return v4_to_current(v4, duration)


def v4_to_current(v4_edit: dict[str, Any], duration: float = 120.0) -> dict[str, Any]:
    return reel_v5.edit_v4_to_v5(v4_edit, RALLIES, duration)


RALLIES = {1: (10.0, 20.0), 2: (30.0, 45.0)}


def v1(*clips: dict[str, Any], **rest: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "view": "full",
        "format": "9:16",
        "clips": list(clips),
        "restored_gaps": [{"start": 20.5, "end": 24.0}],
        "overlays": {"trail": True, "captions": [{"text": "x", "start": 1, "end": 2}]},
        **rest,
    }


def trim(
    rally_id: int | None, start: float, end: float, slow_mo: bool = False
) -> dict[str, Any]:
    return {
        "rally_id": rally_id,
        "trim_start": start,
        "trim_end": end,
        "slow_mo": slow_mo,
    }


def test_each_trim_becomes_one_piece() -> None:
    new = reel_v2.edit_v1_to_v2(
        v1(trim(2, 30, 44, slow_mo=True), trim(1, 9.5, 20.5)), RALLIES
    )
    assert new == {
        "version": 2,
        "view": "full",
        "format": "9:16",
        "lead_in": 1.0,
        "lead_out": 1.0,
        "clips": [
            {"rally_id": 2, "pieces": [{"start": 30, "end": 44, "slow_mo": True}]},
            {"rally_id": 1, "pieces": [{"start": 9.5, "end": 20.5, "slow_mo": False}]},
        ],
        "restored_gaps": [{"start": 20.5, "end": 24.0}],
        "overlays": {"trail": True, "captions": [{"text": "x", "start": 1, "end": 2}]},
    }
    ReelEdit.model_validate(to_current(new))  # valid after the later migrations


def test_lead_grows_to_fit_old_trims_up_to_3_seconds() -> None:
    new = reel_v2.edit_v1_to_v2(v1(trim(1, 8.27, 21.5), trim(2, 25, 52)), RALLIES)
    assert (new["lead_in"], new["lead_out"]) == (3.0, 3.0)
    assert new["clips"][0]["pieces"][0] == {
        "start": 8.27,
        "end": 21.5,
        "slow_mo": False,
    }
    # 5 s before / 7 s after rally 2: cut to its bounds +- 3 s
    assert new["clips"][1]["pieces"][0] == {"start": 27, "end": 48, "slow_mo": False}

    new = reel_v2.edit_v1_to_v2(v1(trim(1, 8.27, 20)), RALLIES)
    assert new["lead_in"] == 1.73  # 10 - 8.27, rounded up to 0.01 s


def test_clips_without_a_rally_are_dropped() -> None:
    """v1 allowed rally_id null (rally gone after re-analysis); v2 needs a rally."""
    new = reel_v2.edit_v1_to_v2(
        v1(trim(None, 50, 55), trim(99, 1, 2), trim(1, 12, 14)), RALLIES
    )
    assert [c["rally_id"] for c in new["clips"]] == [1]


def test_trim_entirely_outside_its_padded_rally_is_dropped() -> None:
    new = reel_v2.edit_v1_to_v2(v1(trim(1, 60, 65)), RALLIES)
    assert new["clips"] == []
    assert new["lead_out"] == 3.0


def test_defaults_for_missing_fields_and_v2_is_left_alone() -> None:
    new = reel_v2.edit_v1_to_v2({"version": 1, "clips": []}, RALLIES)
    assert ReelEdit.model_validate(to_current(new)) == ReelEdit()
    already = ReelEdit().model_dump(mode="json")  # v3: not touched either
    assert reel_v2.edit_v1_to_v2(already, RALLIES) is already


def test_downgrade_turns_pieces_back_into_one_trim() -> None:
    v2 = {
        "version": 2,
        "view": "cut",
        "format": "16:9",
        "lead_in": 1.0,
        "lead_out": 1.0,
        "clips": [
            {"rally_id": 1, "pieces": [
                {"start": 10, "end": 12, "slow_mo": False},
                {"start": 15, "end": 18, "slow_mo": True},
            ]},
            {"rally_id": 2, "pieces": []},
        ],
        "restored_gaps": [],
        "overlays": {"trail": False, "captions": []},
    }  # fmt: skip
    assert reel_v2.edit_v2_to_v1(v2) == {
        "version": 1,
        "view": "cut",
        "format": "16:9",
        "clips": [{"rally_id": 1, "trim_start": 10, "trim_end": 18, "slow_mo": True}],
        "restored_gaps": [],
        "overlays": {"trail": False, "captions": []},
    }


def test_round_trip_of_a_plain_v1_reel() -> None:
    old = v1(trim(2, 30, 44), trim(1, 9.5, 20.5, slow_mo=True))
    assert reel_v2.edit_v2_to_v1(reel_v2.edit_v1_to_v2(old, RALLIES)) == old


def test_one_reel_per_video_keeps_the_most_recently_updated() -> None:
    from datetime import UTC, datetime
    from uuid import UUID

    def t(hour: int) -> datetime:
        return datetime(2026, 11, 12, hour, tzinfo=UTC)

    v1, v2, v3 = (UUID(int=n) for n in (1, 2, 3))
    r = [UUID(int=100 + n) for n in range(6)]
    live = [
        (r[0], v1, t(10), t(9)),
        (r[1], v1, t(12), t(8)),  # v1: most recently updated -> kept
        (r[2], v1, t(11), t(11)),
        (r[3], v2, t(10), t(9)),  # v2: the only one -> kept
        (r[4], v3, t(10), t(9)),
        (r[5], v3, t(10), t(10)),  # v3: same updated_at -> the newer one is kept
    ]
    assert sorted(one_reel.reels_to_retire(live)) == sorted([r[0], r[2], r[4]])
    assert one_reel.reels_to_retire([]) == []


# --- v2 -> v3: restored breaks become anchors ------------------------------------------


def v2(*gaps: tuple[float, float], **rest: Any) -> dict[str, Any]:
    return {
        "version": 2,
        "view": "cut",
        "format": "9:16",
        "lead_in": 1.0,
        "lead_out": 1.0,
        "clips": [
            {"rally_id": 1, "pieces": [{"start": 9, "end": 21, "slow_mo": False}]},
            {"rally_id": 2, "pieces": [{"start": 29, "end": 46, "slow_mo": True}]},
        ],
        "restored_gaps": [{"start": s, "end": e} for s, e in gaps],
        "overlays": {"trail": False, "captions": [{"text": "x", "start": 1, "end": 2}]},
        **rest,
    }


def test_each_restored_range_becomes_its_midpoint() -> None:
    new = reel_v3.edit_v2_to_v3(v2((50, 60), (21, 29), (0, 8.335)))
    expected = {k: v for k, v in v2().items() if k != "restored_gaps"}
    assert new == {**expected, "version": 3, "restored_anchors": [4.168, 25, 55]}
    ReelEdit.model_validate(v4_to_current(reel_v4.edit_v3_to_v4(new, RALLIES, 120.0)))


def test_v3_is_left_alone() -> None:
    already = ReelEdit().model_dump(mode="json")
    assert reel_v3.edit_v2_to_v3(already) is already
    assert reel_v3.edit_v3_to_v2(v2(), 60.0) == v2()  # v2 isn't downgraded again


def test_downgrade_turns_anchors_into_the_ranges_they_restore() -> None:
    """The v3 rule decides: the whole hidden stretch around the anchor (even if it
    grew since); anchors inside a piece restore nothing and are dropped."""
    v3 = reel_v3.edit_v2_to_v3(v2((22, 26), (49, 55)))
    v3["restored_anchors"].append(15)  # inside rally 1's piece
    assert reel_v3.edit_v3_to_v2(v3, 60.0)["restored_gaps"] == [
        {"start": 21, "end": 29},
        {"start": 46, "end": 60},
    ]


def test_round_trip_of_ranges_that_fill_their_stretch() -> None:
    old = v2((0, 9), (21, 29), (46, 60))
    assert reel_v3.edit_v3_to_v2(reel_v3.edit_v2_to_v3(old), 60.0) == old


def test_downgrade_without_a_duration_leaves_out_the_last_stretch() -> None:
    v3 = reel_v3.edit_v2_to_v3(v2((22, 26), (49, 55)))
    assert reel_v3.edit_v3_to_v2(v3, None)["restored_gaps"] == [
        {"start": 21, "end": 29}
    ]


# --- v3 -> v4: deleted footage becomes cuts ----------------------------------------------


def v3(*clips: dict[str, Any], lead: float = 1.0) -> dict[str, Any]:
    return {
        "version": 3,
        "view": "cut",
        "format": "16:9",
        "lead_in": lead,
        "lead_out": lead,
        "clips": list(clips),
        "restored_anchors": [25.0],
        "overlays": {"trail": False, "captions": []},
    }


def pieces(rally_id: int, *spans: tuple[float, float]) -> dict[str, Any]:
    return {
        "rally_id": rally_id,
        "pieces": [{"start": s, "end": e, "slow_mo": False} for s, e in spans],
    }


def cuts_of(edit: dict[str, Any]) -> list[list[tuple[float, float]]]:
    return [[(c["start"], c["end"]) for c in clip["cuts"]] for clip in edit["clips"]]


def test_window_parts_no_piece_covers_become_cuts() -> None:
    """Rallies 10-20 and 30-45, lead 1 s: windows 9-21 and 29-46."""
    old = v3(pieces(1, (10, 14), (16, 21)), pieces(2, (29, 40)))
    new = reel_v4.edit_v3_to_v4(old, RALLIES, 120.0)
    assert new["version"] == 4
    assert cuts_of(new) == [[(9, 10), (14, 16)], [(40, 46)]]
    assert new["restored_anchors"] == [25.0]
    ReelEdit.model_validate(v4_to_current(new))


def test_windows_like_the_editor() -> None:
    """Close rallies share the gap between them in proportion to the leads (rally 1
    ends at 20, rally 2 starts at 21: the boundary is 20.5 with equal leads); the end
    of the video caps a window; slivers under 0.05 s aren't cuts."""
    rallies = {1: (10.0, 20.0), 2: (21.0, 30.0)}
    windows = reel_v4.rally_windows(rallies, 1.0, 1.0, 30.5)
    assert windows == {1: (9.0, 20.5), 2: (20.5, 30.5)}
    windows = reel_v4.rally_windows(rallies, 0.5, 1.5, None)
    assert windows[1] == (9.5, 20.75)  # 20 + 1 * 1.5 / 2
    new = reel_v4.edit_v3_to_v4(
        v3(pieces(1, (9.02, 20)), pieces(2, (21, 30))), rallies, 30.5
    )
    assert cuts_of(new) == [[(20, 20.5)], [(20.5, 21), (30, 30.5)]]


def test_window_is_widened_to_its_pieces_and_replays_dont_repeat_cuts() -> None:
    """A piece past its window (saved elsewhere) widens it; a rally used twice gives
    its cuts to its first clip; what another clip's piece covers isn't cut."""
    old = v3(pieces(1, (8.5, 15)), pieces(2, (29, 46)), pieces(1, (17, 19)))
    new = reel_v4.edit_v3_to_v4(old, RALLIES, 120.0)
    assert cuts_of(new) == [[(15, 17), (19, 21)], [], []]


def test_clip_of_a_missing_rally_gets_no_cuts() -> None:
    new = reel_v4.edit_v3_to_v4(v3(pieces(99, (1, 2))), RALLIES, 120.0)
    assert cuts_of(new) == [[]]


def test_v4_downgrade_drops_cuts_and_round_trips() -> None:
    old = v3(pieces(1, (10, 14)), pieces(2, (29, 46)))
    new = reel_v4.edit_v3_to_v4(old, RALLIES, 120.0)
    assert reel_v4.edit_v4_to_v3(new) == old
    assert reel_v4.edit_v3_to_v4(new, RALLIES, 120.0) is new  # v4 left alone
    assert reel_v4.edit_v4_to_v3(old) is old


# --- v4 -> v5: the segments v4 played -----------------------------------------------

EDIT_V4 = Path(__file__).resolve().parent / "fixtures" / "edit_v4"


def v4_cases() -> list[Path]:
    files = sorted(EDIT_V4.glob("*.json"))
    assert files, f"no v4 cases in {EDIT_V4}"
    return files


@pytest.mark.parametrize("path", v4_cases(), ids=lambda p: p.stem)
def test_v4_reels_become_the_segments_they_played(path: Path) -> None:
    """Each v4 reel plays the same as v5: the segments its Rallies-only playback
    played, in order (restored stretches unlabelled), and its export has the same
    captions and length."""
    case = json.loads(path.read_text(encoding="utf-8"))
    rallies = {r["id"]: (r["start"], r["end"]) for r in case["rallies"]}
    new = reel_v5.edit_v4_to_v5(case["edit"], rallies, case["duration"])
    edit = ReelEdit.model_validate(new)
    expected = case["expected"]

    assert [(s.start, s.end, s.slow_mo) for s in edit.segments] == [
        (e["start"], e["end"], e["slow_mo"]) for e in expected["segments"]
    ]
    assert [s.rally_id is None for s in edit.segments] == [
        e["kind"] == "gap" for e in expected["segments"]
    ]
    snapshot = build_snapshot(edit, duration=case["duration"])
    assert snapshot.model_dump()["captions"] == expected["captions"]
    assert snapshot.duration == expected["duration"]
    assert new["format"] == case["edit"]["format"]


def test_pieces_keep_their_rally_label_and_ids_are_in_order() -> None:
    old = v3(pieces(2, (29, 46)), pieces(1, (9, 21)), pieces(99, (50, 52)))
    old["restored_anchors"] = []
    v4 = reel_v4.edit_v3_to_v4(old, RALLIES, 120.0)
    new = reel_v5.edit_v4_to_v5(v4, RALLIES, 120.0)
    assert new == {
        "version": 5,
        "format": "16:9",
        "segments": [
            {"id": "s1", "start": 29, "end": 46, "slow_mo": False, "rally_id": 2},
            {"id": "s2", "start": 9, "end": 21, "slow_mo": False, "rally_id": 1},
            # rally 99 doesn't exist (any more): still plays, unlabelled
            {"id": "s3", "start": 50, "end": 52, "slow_mo": False, "rally_id": None},
        ],
        "captions": [],
    }


def test_times_go_to_tenths_and_slivers_join_a_neighbour() -> None:
    """v5 keeps 0.1 s and needs 0.3 s: 12.04 -> 12.0, 20.46 -> 20.5; a 0.2 s piece
    touching one of the same speed joins it, one on its own is dropped."""
    old = {
        "version": 4,
        "view": "cut",
        "format": "9:16",
        "lead_in": 1.0,
        "lead_out": 1.0,
        "clips": [
            {"rally_id": 1, "pieces": [
                {"start": 9.0, "end": 12.04, "slow_mo": False},
                {"start": 12.04, "end": 12.24, "slow_mo": False},  # joins the first
                {"start": 15.0, "end": 20.46, "slow_mo": True},
            ], "cuts": []},
            {"rally_id": 2, "pieces": [
                {"start": 29.5, "end": 29.7, "slow_mo": False},  # alone: dropped
                {"start": 30.0, "end": 30.2, "slow_mo": True},  # joins the next
                {"start": 30.2, "end": 40.0, "slow_mo": True},
            ], "cuts": []},
        ],
        "restored_anchors": [],
        "overlays": {"trail": True, "captions": [{"text": "x", "start": 1, "end": 2}]},
    }  # fmt: skip
    new = reel_v5.edit_v4_to_v5(old, RALLIES, 120.0)
    assert [(s["start"], s["end"], s["slow_mo"]) for s in new["segments"]] == [
        (9.0, 12.2, False),
        (15.0, 20.5, True),
        (30.0, 40.0, True),
    ]
    assert [s["id"] for s in new["segments"]] == ["s1", "s2", "s3"]
    assert new["captions"] == [{"text": "x", "start": 1, "end": 2}]
    assert new["format"] == "9:16"
    ReelEdit.model_validate(new)


def test_v5_is_left_alone() -> None:
    already = ReelEdit().model_dump(mode="json")
    assert reel_v5.edit_v4_to_v5(already, RALLIES, 120.0) is already
    assert (
        reel_v5.edit_v5_to_v4(load_case("01")["edit"], RALLIES)
        == load_case("01")["edit"]
    )


def load_case(prefix: str) -> dict[str, Any]:
    (path,) = EDIT_V4.glob(f"{prefix}-*.json")
    return dict(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("prefix", ["01", "03", "04"])
def test_downgrade_round_trips_plain_reels(prefix: str) -> None:
    """Rally clips with 1 s leads and restored breaks between them come back as
    they were (each break's anchor at its middle)."""
    case = load_case(prefix)
    rallies = {r["id"]: (r["start"], r["end"]) for r in case["rallies"]}
    v5 = reel_v5.edit_v4_to_v5(case["edit"], rallies, case["duration"])
    assert reel_v5.edit_v5_to_v4(v5, rallies) == case["edit"]


def test_downgrade_where_possible() -> None:
    """A replay of the same rally starts another clip; leads grow to fit (up to
    3 s, further is cut off); footage of a deleted rally becomes a restored anchor;
    captions go back to overlays."""
    v5 = {
        "version": 5,
        "format": "16:9",
        "segments": [
            {"id": "a", "start": 8.5, "end": 15, "slow_mo": False, "rally_id": 1},
            {"id": "b", "start": 15, "end": 21, "slow_mo": True, "rally_id": 1},
            {"id": "c", "start": 30, "end": 52, "slow_mo": False, "rally_id": 2},
            {"id": "d", "start": 12, "end": 14, "slow_mo": True, "rally_id": 1},
            {"id": "e", "start": 60, "end": 64, "slow_mo": False, "rally_id": 7},
        ],
        "captions": [{"text": "x", "start": 1, "end": 2}],
    }
    old = reel_v5.edit_v5_to_v4(v5, RALLIES)
    assert (old["version"], old["lead_in"], old["lead_out"]) == (4, 1.5, 3.0)
    assert [(c["rally_id"], [(p["start"], p["end"]) for p in c["pieces"]]) for c in old["clips"]] == [
        (1, [(8.5, 15), (15, 21)]),
        (2, [(30, 48)]),  # 52 is past rally 2's end + 3 s
        (1, [(12, 14)]),  # a replay: its own clip
    ]  # fmt: skip
    assert old["restored_anchors"] == [62]
    assert old["overlays"] == {"trail": False, "captions": v5["captions"]}
    assert all(c["cuts"] == [] for c in old["clips"])
