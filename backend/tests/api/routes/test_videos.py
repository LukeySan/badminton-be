import subprocess
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app import crud, media
from app.core.config import settings
from app.core.security import create_access_token, create_file_token
from app.models import Rally, UserCreate, Video
from tests.utils.user import user_authentication_headers
from tests.utils.utils import random_email, random_lower_string

API = f"{settings.API_V1_STR}/videos"
COURT = {"corners": [[0.478, 0.347], [0.777, 0.369], [0.93, 0.943], [0.054, 0.781]]}


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 3 s, 320x240, 30 fps test pattern made with ffmpeg."""
    path = tmp_path_factory.mktemp("videos") / "sample.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=duration=3:size=320x240:rate=30", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )  # fmt: skip
    return path


@pytest.fixture(scope="module")
def rotated_video(sample_video: Path) -> Path:
    """Same pixels, but with phone-style rotation metadata (displayed as portrait)."""
    path = sample_video.with_name("rotated.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-display_rotation", "90",
         "-i", str(sample_video), "-c", "copy", str(path)],
        check=True,
    )  # fmt: skip
    return path


@pytest.fixture(scope="module")
def other_user_headers(client: TestClient, db: Session) -> dict[str, str]:
    email, password = random_email(), random_lower_string()
    crud.create_user(session=db, user_create=UserCreate(email=email, password=password))
    return user_authentication_headers(client=client, email=email, password=password)


def upload(
    client: TestClient, headers: dict[str, str], path: Path, name: str | None = None
) -> httpx.Response:
    with open(path, "rb") as f:
        return client.post(
            f"{API}/", headers=headers, files={"file": (name or path.name, f, "video/mp4")}
        )


@pytest.fixture
def video(
    client: TestClient, normal_user_token_headers: dict[str, str], sample_video: Path
) -> dict[str, Any]:
    r = upload(client, normal_user_token_headers, sample_video)
    assert r.status_code == 201, r.text
    data: dict[str, Any] = r.json()
    return data


def stored_videos() -> set[str]:
    return {p.name for p in settings.STORAGE_DIR.iterdir()}


# --- upload ---------------------------------------------------------------


def test_upload_video(video: dict[str, Any]) -> None:
    assert video["status"] == "uploaded"
    assert video["filename"] == "sample.mp4"
    assert (video["width"], video["height"], video["fps"]) == (320, 240, 30.0)
    assert video["duration"] == pytest.approx(3.0, abs=0.1)
    assert video["court_corners"] is None
    assert (settings.STORAGE_DIR / video["id"] / "original.mp4").exists()


def test_upload_rotated_video_reports_displayed_size(
    client: TestClient, normal_user_token_headers: dict[str, str], rotated_video: Path
) -> None:
    r = upload(client, normal_user_token_headers, rotated_video)
    assert r.status_code == 201
    assert (r.json()["width"], r.json()["height"]) == (240, 320)


def test_upload_requires_login(client: TestClient, sample_video: Path) -> None:
    r = upload(client, {}, sample_video)
    assert r.status_code == 401


def test_upload_unsupported_extension(
    client: TestClient, normal_user_token_headers: dict[str, str], sample_video: Path
) -> None:
    before = stored_videos()
    r = upload(client, normal_user_token_headers, sample_video, name="notes.txt")
    assert r.status_code == 415
    assert stored_videos() == before


def test_upload_not_a_video_is_rejected_and_cleaned_up(
    client: TestClient, normal_user_token_headers: dict[str, str], tmp_path: Path
) -> None:
    fake = tmp_path / "fake.mp4"
    fake.write_text("not a video")
    before = stored_videos()
    r = upload(client, normal_user_token_headers, fake)
    assert r.status_code == 400
    assert "not a readable video" in r.json()["detail"]
    assert stored_videos() == before


# --- ownership --------------------------------------------------------------


def test_list_shows_only_own_videos(
    client: TestClient,
    video: dict[str, Any],
    normal_user_token_headers: dict[str, str],
    other_user_headers: dict[str, str],
    superuser_token_headers: dict[str, str],
) -> None:
    def ids(headers: dict[str, str]) -> set[str]:
        r = client.get(f"{API}/", headers=headers)
        assert r.status_code == 200
        assert r.json()["count"] >= len(r.json()["data"])
        return {v["id"] for v in r.json()["data"]}

    assert video["id"] in ids(normal_user_token_headers)
    assert video["id"] not in ids(other_user_headers)
    assert video["id"] in ids(superuser_token_headers)


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", ""),
        ("get", "/frame"),
        ("put", "/court"),
        ("post", "/process"),
        ("get", "/rallies"),
        ("get", "/file-url"),
    ],
)
def test_other_users_video_is_not_found(
    client: TestClient,
    video: dict[str, Any],
    other_user_headers: dict[str, str],
    method: str,
    path: str,
) -> None:
    kwargs: dict[str, Any] = {"json": COURT} if method == "put" else {}
    r = client.request(
        method, f"{API}/{video['id']}{path}", headers=other_user_headers, **kwargs
    )
    assert r.status_code == 404


def test_missing_video_is_not_found(
    client: TestClient, normal_user_token_headers: dict[str, str]
) -> None:
    r = client.get(f"{API}/{uuid.uuid4()}", headers=normal_user_token_headers)
    assert r.status_code == 404


def test_superuser_can_read_any_video(
    client: TestClient, video: dict[str, Any], superuser_token_headers: dict[str, str]
) -> None:
    r = client.get(f"{API}/{video['id']}", headers=superuser_token_headers)
    assert r.status_code == 200


# --- court + status changes -------------------------------------------------


@pytest.mark.parametrize(
    ("corners", "message"),
    [
        ([[0.1, 0.3], [0.9, 0.3], [0.9, 0.9]], "at least 4"),
        ([[0.1, 0.3], [0.9, 0.3], [1.2, 0.9], [0.1, 0.9]], "less than or equal to 1"),
        ([[0.1, 0.9], [0.9, 0.9], [0.9, 0.3], [0.1, 0.3]], "far corners must be above"),
        ([[0.9, 0.3], [0.1, 0.3], [0.9, 0.9], [0.1, 0.9]], "left corners must be left"),
    ],
)
def test_court_validation(
    client: TestClient,
    video: dict[str, Any],
    normal_user_token_headers: dict[str, str],
    corners: list[list[float]],
    message: str,
) -> None:
    r = client.put(
        f"{API}/{video['id']}/court",
        headers=normal_user_token_headers,
        json={"corners": corners},
    )
    assert r.status_code == 422
    assert message in r.text


def test_court_and_process_flow(
    client: TestClient, video: dict[str, Any], normal_user_token_headers: dict[str, str]
) -> None:
    url = f"{API}/{video['id']}"
    h = normal_user_token_headers

    r = client.post(f"{url}/process", headers=h)
    assert r.status_code == 409
    assert "court corners first" in r.json()["detail"]

    r = client.put(f"{url}/court", headers=h, json=COURT)
    assert r.status_code == 200
    assert r.json()["court_corners"] == COURT["corners"]

    r = client.post(f"{url}/process", headers=h)
    assert r.status_code == 200
    assert r.json()["status"] == "queued"

    r = client.post(f"{url}/process", headers=h)
    assert r.status_code == 409
    assert r.json()["detail"] == "Video is already queued"

    r = client.put(f"{url}/court", headers=h, json=COURT)
    assert r.status_code == 409
    assert "while the video is queued" in r.json()["detail"]


def test_done_video_can_be_reprocessed(
    client: TestClient,
    db: Session,
    video: dict[str, Any],
    normal_user_token_headers: dict[str, str],
) -> None:
    row = db.get(Video, uuid.UUID(video["id"]))
    assert row
    row.status, row.court_corners = "done", COURT["corners"]
    db.add(row)
    db.commit()

    r = client.post(f"{API}/{video['id']}/process", headers=normal_user_token_headers)
    assert r.status_code == 200
    assert r.json()["status"] == "queued"


# --- frame, rallies, file ---------------------------------------------------


def test_frame(
    client: TestClient, video: dict[str, Any], normal_user_token_headers: dict[str, str]
) -> None:
    for t in (1.0, 9999.0):  # past the end is clamped to the last frames
        r = client.get(
            f"{API}/{video['id']}/frame",
            headers=normal_user_token_headers,
            params={"t": t},
        )
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert r.content[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_frame_error_is_reported(
    client: TestClient,
    video: dict[str, Any],
    normal_user_token_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_args: Any) -> bytes:
        raise RuntimeError("decoder exploded")

    monkeypatch.setattr(media, "frame_jpeg", broken)
    r = client.get(f"{API}/{video['id']}/frame", headers=normal_user_token_headers)
    assert r.status_code == 500
    assert "decoder exploded" in r.json()["detail"]


def test_rallies_sorted_with_duration(
    client: TestClient,
    db: Session,
    video: dict[str, Any],
    normal_user_token_headers: dict[str, str],
) -> None:
    vid = uuid.UUID(video["id"])
    db.add_all([Rally(video_id=vid, start=20.5, end=31.0), Rally(video_id=vid, start=3.0, end=9.25)])
    db.commit()

    r = client.get(f"{API}/{video['id']}/rallies", headers=normal_user_token_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert [(x["start"], x["end"], x["duration"]) for x in body["data"]] == [
        (3.0, 9.25, 6.25),
        (20.5, 31.0, 10.5),
    ]


def test_file_url_streams_with_range(
    client: TestClient, video: dict[str, Any], normal_user_token_headers: dict[str, str]
) -> None:
    r = client.get(f"{API}/{video['id']}/file-url", headers=normal_user_token_headers)
    assert r.status_code == 200
    url = r.json()["url"]
    assert url.startswith(f"{API}/{video['id']}/file?token=")

    # The browser's <video> seeks with Range requests; they must get 206
    r = client.get(url, headers={"Range": "bytes=0-99"})
    assert r.status_code == 206
    assert r.headers["content-range"].startswith("bytes 0-99/")
    assert len(r.content) == 100


def test_file_rejects_bad_tokens(
    client: TestClient, video: dict[str, Any], normal_user_token_headers: dict[str, str]
) -> None:
    file_url = f"{API}/{video['id']}/file"
    other_video_token, _ = create_file_token(uuid.uuid4())
    login_token = normal_user_token_headers["Authorization"].removeprefix("Bearer ")

    assert client.get(file_url).status_code == 422  # no token
    for token in ("garbage", other_video_token, login_token):
        assert client.get(file_url, params={"token": token}).status_code == 403


def test_file_token_is_not_a_login_token(
    client: TestClient, video: dict[str, Any]
) -> None:
    file_token, _ = create_file_token(uuid.UUID(video["id"]))
    r = client.get(
        f"{settings.API_V1_STR}/users/me",
        headers={"Authorization": f"Bearer {file_token}"},
    )
    assert r.status_code == 403


def test_file_for_deleted_video(client: TestClient) -> None:
    missing = uuid.uuid4()
    token, _ = create_file_token(missing)
    r = client.get(f"{API}/{missing}/file", params={"token": token})
    assert r.status_code == 404


def test_access_token_still_works(client: TestClient, db: Session) -> None:
    """Sanity check for the scope check in get_current_user: plain login tokens pass."""
    user = crud.get_user_by_email(session=db, email=settings.FIRST_SUPERUSER)
    assert user
    token = create_access_token(user.id, timedelta(minutes=5))
    r = client.get(
        f"{settings.API_V1_STR}/users/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200


def test_deleting_account_removes_video_files(
    client: TestClient, db: Session, sample_video: Path
) -> None:
    email, password = random_email(), random_lower_string()
    crud.create_user(session=db, user_create=UserCreate(email=email, password=password))
    headers = user_authentication_headers(client=client, email=email, password=password)
    video_id = upload(client, headers, sample_video).json()["id"]
    assert (settings.STORAGE_DIR / video_id).exists()

    r = client.delete(f"{settings.API_V1_STR}/users/me", headers=headers)
    assert r.status_code == 200
    assert not (settings.STORAGE_DIR / video_id).exists()
    db.expire_all()
    assert db.get(Video, uuid.UUID(video_id)) is None  # row removed by cascade
