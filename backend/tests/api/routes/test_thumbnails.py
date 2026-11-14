"""GET /videos/{id}/thumbnails-url and the signed thumbnail files."""

import json
import uuid
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app import crud, thumbnails
from app.core.config import settings
from app.core.security import create_access_token, create_file_token
from app.models import Video

API = f"{settings.API_V1_STR}/videos"
INDEX = {
    "version": 1,
    "interval": 1.0,
    "count": 150,
    "tile_width": 160,
    "tile_height": 90,
    "columns": 10,
    "rows": 10,
    "sheets": ["sheet_000.jpg", "sheet_001.jpg"],
}


def make_video(db: Session, email: str = settings.EMAIL_TEST_USER) -> uuid.UUID:
    owner = crud.get_user_by_email(session=db, email=email)
    assert owner
    video = Video(
        owner_id=owner.id,
        filename="match.mp4",
        storage_path="x/original.mp4",
        status="done",
        duration=150.0,
    )
    db.add(video)
    db.commit()
    return video.id


@pytest.fixture
def video_id(db: Session) -> uuid.UUID:
    """The normal user's video, with a (fake) thumbnail strip."""
    video_id = make_video(db)
    folder = thumbnails.thumbnails_dir(video_id)
    folder.mkdir(parents=True)
    (folder / "index.json").write_text(json.dumps(INDEX))
    for i, name in enumerate(INDEX["sheets"]):  # type: ignore[arg-type]
        (folder / name).write_bytes(b"\xff\xd8 jpeg %d" % i)
    return video_id


def test_signed_links(
    client: TestClient, normal_user_token_headers: dict[str, str], video_id: uuid.UUID
) -> None:
    r = client.get(
        f"{API}/{video_id}/thumbnails-url", headers=normal_user_token_headers
    )
    assert r.status_code == 200, r.text
    urls = r.json()
    assert urls["index_url"].startswith(
        f"{API}/{video_id}/thumbnails/index.json?token="
    )
    assert len(urls["sheet_urls"]) == 2
    assert "/thumbnails/sheet_001.jpg?token=" in urls["sheet_urls"][1]
    assert urls["expires_at"]

    # The links work without the login header (like <img src>)
    r = client.get(urls["index_url"])
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/json"
    assert r.json() == INDEX
    r = client.get(urls["sheet_urls"][1])
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == b"\xff\xd8 jpeg 1"
    assert r.headers["cache-control"] == "private, max-age=3600"


def test_missing_strip_has_a_code(
    client: TestClient, db: Session, normal_user_token_headers: dict[str, str]
) -> None:
    video_id = make_video(db)
    r = client.get(
        f"{API}/{video_id}/thumbnails-url", headers=normal_user_token_headers
    )
    assert r.status_code == 404
    assert r.json()["code"] == "thumbnails_missing"


def test_other_users_and_missing_videos_get_a_plain_404(
    client: TestClient,
    db: Session,
    normal_user_token_headers: dict[str, str],
) -> None:
    theirs = make_video(db, email=settings.FIRST_SUPERUSER)
    for video_id in (theirs, uuid.uuid4()):
        r = client.get(
            f"{API}/{video_id}/thumbnails-url", headers=normal_user_token_headers
        )
        assert r.status_code == 404
        assert r.json() == {"detail": "Video not found"}


def test_needs_login(client: TestClient, video_id: uuid.UUID) -> None:
    assert client.get(f"{API}/{video_id}/thumbnails-url").status_code == 401


def test_superuser_can_get_links(
    client: TestClient, superuser_token_headers: dict[str, str], video_id: uuid.UUID
) -> None:
    r = client.get(f"{API}/{video_id}/thumbnails-url", headers=superuser_token_headers)
    assert r.status_code == 200


def test_files_need_a_valid_token_for_that_video(
    client: TestClient, db: Session, video_id: uuid.UUID
) -> None:
    other_video = make_video(db)
    base = f"{API}/{video_id}/thumbnails/index.json"
    token_for_other, _ = create_file_token(other_video)
    login_token = create_access_token(uuid.uuid4(), timedelta(minutes=5))
    for token in ["nonsense", token_for_other, login_token]:
        assert client.get(f"{base}?token={token}").status_code == 403
    assert client.get(base).status_code == 422  # no token at all


@pytest.mark.parametrize(
    "name",
    ["original.mp4", "..%2Foriginal.mp4", "sheet_1.jpg", "sheet_000.png", "index.json.bak", "sheet_005.jpg"],
)  # fmt: skip
def test_only_strip_files_are_served(
    client: TestClient, video_id: uuid.UUID, name: str
) -> None:
    """Only index.json and sheet_NNN.jpg, and only if they exist: a crafted name
    can't reach the original video or anything outside the folder."""
    (settings.STORAGE_DIR / str(video_id) / "original.mp4").write_bytes(b"video")
    token, _ = create_file_token(video_id)
    r = client.get(f"{API}/{video_id}/thumbnails/{name}?token={token}")
    assert r.status_code == 404
    assert b"video" not in r.content


def test_deleted_video_strip_is_gone(
    client: TestClient, normal_user_token_headers: dict[str, str], video_id: uuid.UUID
) -> None:
    token, _ = create_file_token(video_id)
    r = client.delete(f"{API}/{video_id}", headers=normal_user_token_headers)
    assert r.status_code == 200
    # Even with a still-valid link
    r = client.get(f"{API}/{video_id}/thumbnails/index.json?token={token}")
    assert r.status_code == 404
