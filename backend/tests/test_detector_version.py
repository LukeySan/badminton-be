import pytest

from app.core.config import settings
from app.models import VideoPublic, detector_release


def test_detector_release() -> None:
    assert detector_release("v1") == (1,)
    assert detector_release("v1.1-2026-11") == (1, 1)
    assert detector_release("v2.10") == (2, 10)
    assert detector_release("test-fixture") is None


@pytest.mark.parametrize(
    ("made_with", "current", "outdated"),
    [
        ("v1", "v1.1", True),
        ("v1.1-2026-11", "v1.1", False),
        (None, "v1.1", False),  # not processed yet
        ("test-fixture", "v1.1", False),
        ("v1.1-2026-11", "v1", False),  # rolled back: re-analyzing would be worse
    ],
)
def test_detector_outdated(
    made_with: str | None,
    current: str,
    outdated: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "DETECTOR_VERSION", current)
    video = VideoPublic.model_validate(
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "owner_id": "00000000-0000-0000-0000-000000000002",
            "filename": "a.mp4",
            "status": "done",
            "progress": 1.0,
            "failure_reason": None,
            "duration": 60.0,
            "fps": 30.0,
            "width": 1280,
            "height": 720,
            "court_corners": None,
            "detector_version": made_with,
            "created_at": "2026-11-14T00:00:00Z",
            "updated_at": "2026-11-14T00:00:00Z",
            "rally_count": 3,
            "play_seconds": 20.0,
        }
    )
    assert video.detector_outdated is outdated
