"""
The v1.1 detector (detector.py + rally_features.py + models/v1.1-2026-11/). Needs the
worker's dependencies; run locally from the repo root: uv run pytest worker/tests -q
"""

import json
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKER))

import detector  # noqa: E402
from detector import DETECTORS, detect_rallies, load_model  # noqa: E402

VERSION = DETECTORS["v1.1"]
MODEL_DIR = WORKER / "models" / VERSION
RESEARCH = WORKER.parents[1] / "badminton-test" / "TrackNetV3"
REFERENCES = sorted((MODEL_DIR / "reference").glob("*.json"))


def test_model_folder_matches_its_version() -> None:
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    assert meta["version"] == VERSION
    assert (meta["maybe_at"], meta["accept_at"]) == (0.2, 0.8)
    assert meta["candidate_rule_settings"] == {"still": 0.25, "gap": 2.5, "min_length": 1.0,
                                               "min_speed": 0.004}


def test_model_loads_once_per_process() -> None:
    assert load_model(VERSION) is load_model(VERSION)


def test_refuses_models_from_another_scikit_learn(monkeypatch: pytest.MonkeyPatch) -> None:
    load_model.cache_clear()
    monkeypatch.setattr(detector.sklearn, "__version__", "0.0.0")
    with pytest.raises(RuntimeError, match="pin scikit-learn=="):
        load_model(VERSION)
    load_model.cache_clear()


def test_unknown_detector() -> None:
    with pytest.raises(KeyError):
        detect_rallies("ball.csv", None, "court.json", detector="v9", fps=30)


@pytest.mark.parametrize("ref_path", REFERENCES, ids=[p.stem for p in REFERENCES])
def test_same_output_as_research(ref_path: Path) -> None:
    """Same tracking CSV + same model files -> exactly research's v1.1 output
    (train_v11.py writes these references with research's own code)."""
    ref = json.loads(ref_path.read_text())
    csv, court = RESEARCH / ref["csv"], RESEARCH / ref["court"]
    if not csv.exists():
        pytest.skip(f"research files not on this machine ({csv})")
    got = detect_rallies(str(csv), None, str(court), detector="v1.1", fps=ref["fps"])
    assert got["detector_version"] == VERSION
    assert got["rallies"] == ref["rallies"]


def test_maybe_and_confidence_follow_the_cut_offs() -> None:
    ref = json.loads((MODEL_DIR / "reference" / "match5.json").read_text())
    csv, court = RESEARCH / ref["csv"], RESEARCH / ref["court"]
    if not csv.exists():
        pytest.skip("research files not on this machine")
    rallies = detect_rallies(str(csv), None, str(court), detector="v1.1", fps=ref["fps"])["rallies"]
    assert rallies == sorted(rallies, key=lambda r: r["start"])
    for r in rallies:
        assert 0.2 <= r["confidence"] <= 1.0          # below 0.2 is hidden
        assert r["is_maybe"] == (r["confidence"] < 0.8)
        assert r["duration"] == pytest.approx(r["end"] - r["start"], abs=0.011)
