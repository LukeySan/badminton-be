"""
Worker-side failure detection. Needs OpenCV/numpy (the worker's dependencies), which
CI doesn't install, so run these locally from the repo root:

    uv run pytest worker/tests -q
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector import MIN_SHUTTLE_IN_COURT, detect_rallies  # noqa: E402
from job_errors import CourtNotVisible, VideoUnreadable  # noqa: E402
from pipeline import downscale, video_info, write_court_json  # noqa: E402

FRAMES, FPS, W, H = 300, 30, 1280, 720
COURT = [[0.3, 0.4], [0.7, 0.4], [0.9, 0.9], [0.1, 0.9]]  # fractions, as the API stores them
OUTSIDE = (40, 650)  # bottom-left corner: outside the court and the air above it


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("v") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         f"testsrc=duration={FRAMES / FPS}:size={W}x{H}:rate={FPS}", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@pytest.fixture
def court(tmp_path: Path) -> str:
    return write_court_json(COURT, W, H, str(tmp_path / "court.json"))


def write_track(path: Path, in_court_frames: int) -> str:
    """Shuttle CSV: the first `in_court_frames` frames fly across the court (a
    rally), the rest sit outside the court zone."""
    rows = ["Frame,Visibility,X,Y"]
    for f in range(FRAMES):
        if f < in_court_frames:
            x, y = 500 + (f % 30) * 10, 400 + (f % 15) * 15
        else:
            x, y = OUTSIDE
        rows.append(f"{f},1,{x},{y}")
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def test_court_not_visible(video: Path, court: str, tmp_path: Path) -> None:
    csv = write_track(tmp_path / "ball.csv", in_court_frames=0)
    with pytest.raises(CourtNotVisible, match=r"in 0\.0% of frames"):
        detect_rallies(csv, str(video), court)


def test_court_visibility_threshold(video: Path, court: str, tmp_path: Path) -> None:
    just_below = int(FRAMES * MIN_SHUTTLE_IN_COURT) - 3  # 4% of frames
    with pytest.raises(CourtNotVisible):
        detect_rallies(write_track(tmp_path / "a.csv", just_below), str(video), court)
    enough = int(FRAMES * MIN_SHUTTLE_IN_COURT) + 3  # 6% of frames
    detect_rallies(write_track(tmp_path / "b.csv", enough), str(video), court)


def test_shuttle_in_court_finds_a_rally(video: Path, court: str, tmp_path: Path) -> None:
    rallies = detect_rallies(write_track(tmp_path / "ball.csv", 150), str(video), court)
    assert len(rallies) == 1
    assert rallies[0]["start"] == pytest.approx(0.0, abs=0.3)


def test_unreadable_video(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video" * 1000)
    with pytest.raises(VideoUnreadable):
        video_info(str(junk))
    with pytest.raises(VideoUnreadable):  # first step of every job
        downscale(str(junk), str(tmp_path / "out.mp4"))


def test_readable_video(video: Path) -> None:
    info = video_info(str(video))
    assert (info["width"], info["height"], info["fps"]) == (W, H, FPS)
