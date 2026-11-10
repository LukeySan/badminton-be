"""GET /videos/{id}/trajectory. The files are written with the worker's own writer
(worker/trajectory_file.py, standard library only), so these tests also check that
the worker and the API agree on the file format."""

import gzip
import json
import os
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app import crud, trajectory
from app.core.config import settings
from app.models import Video

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "worker"))
import trajectory_file  # noqa: E402

API = f"{settings.API_V1_STR}/videos"
FPS, FRAMES, W, H = 30.0, 90, 1000, 720  # a 3 s video


def make_video(db: Session, email: str = settings.EMAIL_TEST_USER) -> uuid.UUID:
    owner = crud.get_user_by_email(session=db, email=email)
    assert owner
    video = Video(
        owner_id=owner.id,
        filename="match.mp4",
        storage_path="x/original.mp4",
        status="done",
        duration=FRAMES / FPS,
        fps=FPS,
    )
    db.add(video)
    db.commit()
    return video.id


def write_track(
    video_id: uuid.UUID, tmp_path: Path, frames: int = FRAMES, fps: float = FPS
) -> Path:
    """TrackNet-style CSV in pixels -> trajectory file, as the worker does it.
    The shuttle is seen on even frames only, at x = 10 px per frame, y = 360 px."""
    csv_path = tmp_path / "ball.csv"
    rows = ["Frame,Visibility,X,Y"] + [
        f"{f},1,{f * 10},360" if f % 2 == 0 else f"{f},0,0,0" for f in range(frames)
    ]
    csv_path.write_text("\n".join(rows) + "\n")
    out = trajectory.trajectory_path(video_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    trajectory_file.write_trajectory(str(csv_path), W, H, fps, str(out))
    return out


@pytest.fixture
def video_id(db: Session, tmp_path: Path) -> uuid.UUID:
    video_id = make_video(db)
    write_track(video_id, tmp_path)
    return video_id


def get(
    client: TestClient, headers: dict[str, str], video_id: uuid.UUID, query: str
) -> dict[str, object]:
    r = client.get(f"{API}/{video_id}/trajectory?{query}", headers=headers)
    assert r.status_code == 200, r.text
    data: dict[str, object] = r.json()
    return data


def test_points_in_range(
    client: TestClient, normal_user_token_headers: dict[str, str], video_id: uuid.UUID
) -> None:
    data = get(client, normal_user_token_headers, video_id, "start=1&end=2")
    assert (data["fps"], data["start"], data["end"]) == (FPS, 1.0, 2.0)
    points = data["points"]
    assert isinstance(points, list)
    # Frames 30..60 inclusive: both ends of the range count
    assert [p["frame"] for p in points] == list(range(30, 61))
    assert points[0] == {"frame": 30, "t": 1.0, "visible": True, "x": 0.3, "y": 0.5}
    assert points[1] == {
        "frame": 31,
        "t": 1.033,
        "visible": False,
        "x": None,
        "y": None,
    }


def test_range_past_the_end_is_cut_off(
    client: TestClient, normal_user_token_headers: dict[str, str], video_id: uuid.UUID
) -> None:
    data = get(client, normal_user_token_headers, video_id, "start=2.5&end=10")
    points = data["points"]
    assert isinstance(points, list)
    assert [p["frame"] for p in points] == list(range(75, FRAMES))
    data = get(client, normal_user_token_headers, video_id, "start=5&end=6")
    assert data["points"] == []


def test_between_frames(
    client: TestClient, normal_user_token_headers: dict[str, str], video_id: uuid.UUID
) -> None:
    # 0.01-0.05 s holds frame 1 (0.033 s) only
    data = get(client, normal_user_token_headers, video_id, "start=0.01&end=0.05")
    points = data["points"]
    assert isinstance(points, list)
    assert [p["frame"] for p in points] == [1]


@pytest.mark.parametrize(
    "query",
    [
        "start=2&end=1",
        "start=1&end=1",
        "start=-1&end=1",
        "start=0&end=601",
        "start=1",
        "",
    ],
)
def test_invalid_range(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    video_id: uuid.UUID,
    query: str,
) -> None:
    r = client.get(
        f"{API}/{video_id}/trajectory?{query}", headers=normal_user_token_headers
    )
    assert r.status_code == 422


def test_600_seconds_is_allowed(
    client: TestClient, normal_user_token_headers: dict[str, str], video_id: uuid.UUID
) -> None:
    get(client, normal_user_token_headers, video_id, "start=0&end=600")


def test_missing_trajectory_has_a_code(
    client: TestClient, db: Session, normal_user_token_headers: dict[str, str]
) -> None:
    """Videos analyzed before trajectories were saved: the app sees the code and
    offers "Re-analyze to enable shuttle trail"."""
    video_id = make_video(db)
    r = client.get(
        f"{API}/{video_id}/trajectory?start=0&end=1", headers=normal_user_token_headers
    )
    assert r.status_code == 404
    assert r.json()["code"] == "trajectory_missing"
    assert "re-analyze" in r.json()["detail"]


def test_missing_video_has_no_code(
    client: TestClient, normal_user_token_headers: dict[str, str]
) -> None:
    r = client.get(
        f"{API}/{uuid.uuid4()}/trajectory?start=0&end=1",
        headers=normal_user_token_headers,
    )
    assert r.status_code == 404
    assert r.json() == {"detail": "Video not found"}


def test_needs_login(client: TestClient, video_id: uuid.UUID) -> None:
    assert client.get(f"{API}/{video_id}/trajectory?start=0&end=1").status_code == 401


def test_superuser_can_read_any_video(
    client: TestClient, superuser_token_headers: dict[str, str], video_id: uuid.UUID
) -> None:
    get(client, superuser_token_headers, video_id, "start=0&end=1")


def test_someone_elses_video_is_404_without_a_code(
    client: TestClient,
    db: Session,
    normal_user_token_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    video_id = make_video(db, email=settings.FIRST_SUPERUSER)
    write_track(video_id, tmp_path)
    r = client.get(
        f"{API}/{video_id}/trajectory?start=0&end=1", headers=normal_user_token_headers
    )
    assert r.status_code == 404
    assert "code" not in r.json()


def test_deleted_video_is_404(
    client: TestClient, normal_user_token_headers: dict[str, str], video_id: uuid.UUID
) -> None:
    r = client.delete(f"{API}/{video_id}", headers=normal_user_token_headers)
    assert r.status_code == 200
    assert not trajectory.trajectory_path(video_id).exists()  # files go too
    r = client.get(
        f"{API}/{video_id}/trajectory?start=0&end=1", headers=normal_user_token_headers
    )
    assert r.status_code == 404


def test_reanalyzed_trajectory_is_served_not_the_cached_one(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    video_id: uuid.UUID,
    tmp_path: Path,
) -> None:
    get(client, normal_user_token_headers, video_id, "start=0&end=1")  # now cached
    path = write_track(video_id, tmp_path, frames=60, fps=60.0)
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))  # surely newer
    data = get(client, normal_user_token_headers, video_id, "start=0&end=1")
    assert data["fps"] == 60.0


# --- the file format ----------------------------------------------------------


def test_file_format(video_id: uuid.UUID) -> None:
    with gzip.open(trajectory.trajectory_path(video_id), "rt") as fh:
        data = json.load(fh)
    assert data["version"] == 1
    assert data["fps"] == FPS
    assert len(data["visible"]) == len(data["x"]) == len(data["y"]) == FRAMES
    assert data["visible"][:3] == [1, 0, 1]
    assert data["x"][:3] == [0.0, None, 0.02]


def test_frames_missing_from_the_csv_count_as_not_seen(tmp_path: Path) -> None:
    csv_path = tmp_path / "ball.csv"
    csv_path.write_text("Frame,Visibility,X,Y\n0,1,500,360\n3,1,100,72\n")
    out = trajectory_file.write_trajectory(
        str(csv_path), W, H, FPS, str(tmp_path / "t.gz")
    )
    with gzip.open(out, "rt") as fh:
        data = json.load(fh)
    assert data["visible"] == [1, 0, 0, 1]
    assert data["y"] == [0.5, None, None, 0.1]
    assert not Path(out + ".tmp").exists()


def test_unknown_version_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "future.json.gz"
    with gzip.open(path, "wt") as fh:
        json.dump({"version": 2}, fh)
    with pytest.raises(ValueError, match="unknown trajectory version 2"):
        trajectory._read(str(path), 0)


def test_worker_and_api_use_the_same_file_name() -> None:
    assert trajectory_file.FILENAME == trajectory.FILENAME
