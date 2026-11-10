"""
The one place the rest of the app calls to turn a shuttle track into rallies.

Today it runs the hand-written rules (rally_detect.rule_detect, copied unchanged
from research). To try the ML model later, change only this function.
"""

from job_errors import CourtNotVisible
from rally_detect import rule_detect

# Tuned on matches 4-5 in fast tracking mode (see ../CLAUDE.md).
RULE_SETTINGS = dict(margin=40, min_speed=0.005, gap=2.5, still=0.25, still_speed=0.35, min_length=1.5)

# Share of frames with the shuttle inside the court zone (the court plus the air
# above it). Measured: real courts 25-94% (matches 1-5), deliberately wrong courts
# (a box in the ceiling, the neighbouring court) 0-6.4%. Below this, the result
# would be nonsense, so fail with "Court not visible" instead.
MIN_SHUTTLE_IN_COURT = 0.05


def detect_rallies(csv_path, video_path, court_path, **overrides):
    """Return [{start, end, duration}] in seconds, sorted by start.
    Raises CourtNotVisible if the shuttle is almost never inside the court."""
    r = rule_detect(csv_path, video_path, court_path, **{**RULE_SETTINGS, **overrides})
    in_court = float(r["vis"].mean()) if len(r["vis"]) else 0.0
    if in_court < MIN_SHUTTLE_IN_COURT:
        raise CourtNotVisible(
            f"shuttle inside the court zone in {100 * in_court:.1f}% of frames "
            f"(minimum {100 * MIN_SHUTTLE_IN_COURT:.0f}%); seen anywhere in "
            f"{100 * float(r['vis_raw'].mean()) if len(r['vis_raw']) else 0:.1f}%"
        )
    fps = r["fps"]
    return [
        {"start": round(s / fps, 2), "end": round(e / fps, 2), "duration": round((e - s) / fps, 2)}
        for s, e in r["segments"]
    ]
