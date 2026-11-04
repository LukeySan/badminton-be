"""
The one place the rest of the app calls to turn a shuttle track into rallies.

Today it runs the hand-written rules (rally_detect.rule_detect, copied unchanged
from research). To try the ML model later, change only this function.
"""

from rally_detect import rule_detect

# Tuned on matches 4-5 in fast tracking mode (see ../CLAUDE.md).
RULE_SETTINGS = dict(margin=40, min_speed=0.005, gap=2.5, still=0.25, still_speed=0.35, min_length=1.5)


def detect_rallies(csv_path, video_path, court_path, **overrides):
    """Return [{start, end, duration}] in seconds, sorted by start."""
    r = rule_detect(csv_path, video_path, court_path, **{**RULE_SETTINGS, **overrides})
    fps = r["fps"]
    return [
        {"start": round(s / fps, 2), "end": round(e / fps, 2), "duration": round((e - s) / fps, 2)}
        for s, e in r["segments"]
    ]
