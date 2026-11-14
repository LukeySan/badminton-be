"""
The one place the rest of the app calls to turn a shuttle track into rallies.

Two detectors, picked by name (settings.DETECTOR_VERSION in the worker, --detector
in the pipeline CLI); each video records the version that produced its rallies:

  "v1"    the hand-written rules (rally_detect.rule_detect, copied from research).
  "v1.1"  looser rules propose candidate rallies; a segment classifier scores each
          one (shown if p >= 0.8, shown as a "maybe" the user confirms if
          0.2 <= p < 0.8, hidden below); the hybrid frame model sets their
          start/end. Models: models/<version>/, trained in research
          (../badminton-test/TrackNetV3/train_v11.py, EXPERIMENTS.md).

v1 stays available for comparison and rollback.
"""

import functools
import json
import os
from typing import Any, NamedTuple

import cv2
import joblib
import sklearn

from job_errors import CourtNotVisible
from rally_detect import rule_detect
from rally_features import frame_features, refine, segment_features, to_rallies

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Setting value -> the version recorded on the video. A new model is a new folder in
# models/ and a new entry here; old videos keep the version they were made with.
DETECTORS = {"v1": "v1", "v1.1": "v1.1-2026-11"}
DEFAULT_DETECTOR = "v1.1"

# v1: tuned on matches 4-5 in fast tracking mode (see ../CLAUDE.md).
RULE_SETTINGS = dict(margin=40, min_speed=0.005, gap=2.5, still=0.25, still_speed=0.35, min_length=1.5)

# Share of frames with the shuttle inside the court zone (the court plus the air
# above it). Measured: real courts 25-94% (matches 1-5), deliberately wrong courts
# (a box in the ceiling, the neighbouring court) 0-6.4%. Below this, the result
# would be nonsense, so fail with "Court not visible" instead.
MIN_SHUTTLE_IN_COURT = 0.05


class Model(NamedTuple):
    hybrid: Any       # frame model: P(frame is in a rally)
    classifier: Any   # segment model: P(candidate is a real rally)
    meta: dict[str, Any]


@functools.cache
def load_model(version):
    """Load a model folder once per process. Refuses files from another scikit-learn
    version: pickled models can load without an error and still predict differently."""
    folder = os.path.join(MODELS_DIR, version)
    with open(os.path.join(folder, "meta.json")) as fh:
        meta = json.load(fh)
    trained_with = meta["libraries"]["scikit-learn"]
    if sklearn.__version__ != trained_with:
        raise RuntimeError(
            f"{version} was trained with scikit-learn {trained_with}, this worker has "
            f"{sklearn.__version__}: pin scikit-learn=={trained_with} in worker/pyproject.toml"
        )
    return Model(joblib.load(os.path.join(folder, "hybrid.joblib")),
                 joblib.load(os.path.join(folder, "classifier.joblib")), meta)


def preload(detector):
    """Load the detector's model now (at worker start), so a missing or mismatched
    model fails immediately instead of on the first video."""
    if DETECTORS[detector] != "v1":
        load_model(DETECTORS[detector])


def detect_rallies(csv_path, video_path, court_path, detector=DEFAULT_DETECTOR, fps=None):
    """Return {"detector_version", "rallies": [{start, end, duration, confidence, is_maybe}]}
    with times in seconds, sorted by start. confidence (0-1) is the classifier's
    probability that the rally is real; None for v1. fps: the video's frame rate
    (read from video_path when not given).
    Raises CourtNotVisible if the shuttle is almost never inside the court."""
    version = DETECTORS[detector]
    if fps is None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    if version == "v1":
        r = _checked_rules(csv_path, court_path, fps, RULE_SETTINGS)
        rallies = [_rally(s / fps, e / fps, None, False) for s, e in r["segments"]]
    else:
        rallies = _detect_v11(csv_path, court_path, fps, load_model(version))
    return {"detector_version": version, "rallies": rallies}


def _checked_rules(csv_path, court_path, fps, settings):
    r = rule_detect(csv_path, None, court_path, fps=fps, **settings)
    in_court = float(r["vis"].mean()) if len(r["vis"]) else 0.0
    if in_court < MIN_SHUTTLE_IN_COURT:
        raise CourtNotVisible(
            f"shuttle inside the court zone in {100 * in_court:.1f}% of frames "
            f"(minimum {100 * MIN_SHUTTLE_IN_COURT:.0f}%); seen anywhere in "
            f"{100 * float(r['vis_raw'].mean()) if len(r['vis_raw']) else 0:.1f}%"
        )
    return r


def _detect_v11(csv_path, court_path, fps, model):
    meta = model.meta
    candidates = _checked_rules(csv_path, court_path, fps, meta["candidate_rule_settings"])
    if not candidates["segments"]:
        return []
    F = frame_features(csv_path, court_path, fps, meta["hybrid_rule_settings"])
    prob = model.hybrid.predict_proba(F[meta["hybrid_columns"]])[:, 1]
    table, cands = segment_features(F, candidates, prob, fps)
    p = model.classifier.predict_proba(table[meta["classifier_columns"]])[:, 1]
    shown = [i for i in range(len(cands)) if p[i] >= meta["maybe_at"]]
    bounds = refine([cands[i] for i in shown], to_rallies(prob, fps, **meta["to_rallies"]),
                    meta["refine_seconds"])
    rallies = [_rally(s, e, float(p[i]), bool(p[i] < meta["accept_at"]))
               for i, (s, e) in zip(shown, bounds)]
    return sorted(rallies, key=lambda r: r["start"])


def _rally(start, end, confidence, is_maybe):
    """Seconds rounded to 0.01; the duration from the unrounded times (as v1 always did)."""
    return {"start": round(start, 2), "end": round(end, 2), "duration": round(end - start, 2),
            "confidence": None if confidence is None else round(confidence, 4), "is_maybe": is_maybe}
