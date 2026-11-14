"""Data conversions inside migrations (pure functions, loaded from the file)."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

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
    ReelEdit.model_validate(new)  # a valid v2 edit


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
    assert ReelEdit.model_validate(new) == ReelEdit()
    already = ReelEdit().model_dump(mode="json")
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
