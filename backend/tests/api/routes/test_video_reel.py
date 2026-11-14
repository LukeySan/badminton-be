"""One reel per video: GET/PUT /videos/{id}/reel (and the deprecated POST's
get-or-create behaviour that keeps the old frontend working)."""

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app import crud
from app.core.config import settings
from app.main import app
from app.models import Reel, ReelEdit, UserCreate, Video
from tests.api.routes.test_reels import make_video
from tests.utils.user import user_authentication_headers
from tests.utils.utils import random_email, random_lower_string

API = settings.API_V1_STR


@pytest.fixture
def video(db: Session) -> tuple[uuid.UUID, list[int]]:
    """The normal user's video (120 s, rallies 10-20 and 30-45 s), no reel yet."""
    return make_video(db, settings.EMAIL_TEST_USER)


def reel_url(video_id: uuid.UUID) -> str:
    return f"{API}/videos/{video_id}/reel"


def save(
    client: TestClient,
    headers: dict[str, str],
    video_id: uuid.UUID,
    updated_at: str | None,
    title: str = "Highlights",
    edit: dict[str, Any] | None = None,
) -> Any:
    return client.put(
        reel_url(video_id),
        headers=headers,
        json={"title": title, "edit": edit or {}, "updated_at": updated_at},
    )


def clip(rally_id: int, start: float, end: float) -> dict[str, Any]:
    return {
        "rally_id": rally_id,
        "pieces": [{"start": start, "end": end, "slow_mo": False}],
    }


def test_no_reel_yet(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    r = client.get(reel_url(video[0]), headers=normal_user_token_headers)
    assert r.status_code == 404
    assert r.json() == {"detail": "This video has no reel yet", "code": "reel_missing"}


def test_create_then_update(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h, a = normal_user_token_headers, video[1][0]
    r = save(client, h, video[0], None, edit={"clips": [clip(a, 10, 12)]})
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["video_id"] == str(video[0])
    assert created["edit"]["clips"] == [clip(a, 10, 12)]
    assert client.get(reel_url(video[0]), headers=h).json() == created

    r = save(client, h, video[0], created["updated_at"], title="Renamed",
             edit={"format": "9:16", "clips": [clip(a, 11, 19)]})  # fmt: skip
    assert r.status_code == 200, r.text
    updated = r.json()
    assert (updated["id"], updated["created_at"]) == (
        created["id"],
        created["created_at"],
    )
    assert updated["title"] == "Renamed"
    assert updated["edit"]["format"] == "9:16"
    assert updated["edit"]["clips"] == [clip(a, 11, 19)]
    assert updated["updated_at"] != created["updated_at"]
    assert client.get(reel_url(video[0]), headers=h).json() == updated


def test_stale_saves_are_refused(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    created = save(client, h, video[0], None, title="First").json()

    # "Create" when it already exists (another tab created it): refused
    r = save(client, h, video[0], None, title="From another tab")
    assert r.status_code == 409
    assert r.json()["code"] == "reel_changed"

    # Two tabs with the same version: the second one is refused
    assert (
        save(client, h, video[0], created["updated_at"], title="Tab A").status_code
        == 200
    )
    r = save(client, h, video[0], created["updated_at"], title="Tab B")
    assert r.status_code == 409
    assert r.json()["code"] == "reel_changed"
    assert client.get(reel_url(video[0]), headers=h).json()["title"] == "Tab A"


def test_save_of_a_deleted_reel_is_refused_then_a_new_one_can_be_made(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    old = save(client, h, video[0], None).json()
    assert client.delete(f"{API}/reels/{old['id']}", headers=h).status_code == 200

    assert save(client, h, video[0], old["updated_at"]).status_code == 409
    assert client.get(reel_url(video[0]), headers=h).json()["code"] == "reel_missing"
    r = save(client, h, video[0], None, title="Fresh start")
    assert r.status_code == 201
    assert r.json()["id"] != old["id"]


def test_edit_is_validated_and_nothing_is_saved(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h, a = normal_user_token_headers, video[1][0]
    r = save(client, h, video[0], None, edit={"clips": [clip(999_999_999, 1, 2)]})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "edit", "clips", 0, "rally_id"]
    r = save(client, h, video[0], None, edit={"clips": [clip(a, 5, 12)]})  # 1 s lead
    assert r.json()["detail"][0]["loc"] == [
        "body",
        "edit",
        "clips",
        0,
        "pieces",
        0,
        "start",
    ]
    assert client.get(reel_url(video[0]), headers=h).status_code == 404


def test_empty_clips_are_dropped(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    a = video[1][0]
    edit = {"clips": [{"rally_id": a, "pieces": []}, clip(a, 10, 12)]}
    r = save(client, normal_user_token_headers, video[0], None, edit=edit)
    assert r.json()["edit"]["clips"] == [clip(a, 10, 12)]


@pytest.mark.parametrize(
    "body",
    [
        {"edit": {}, "updated_at": None},  # no title
        {"title": "x", "updated_at": None},  # no edit
        {"title": "x", "edit": {}, "updated_at": "2026-11-12T10:00:00"},  # no timezone
        {"title": "x", "edit": {"version": 1}, "updated_at": None},
    ],
)
def test_invalid_body(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    body: dict[str, Any],
) -> None:
    r = client.put(reel_url(video[0]), headers=normal_user_token_headers, json=body)
    assert r.status_code == 422


def test_superuser_saves_it_for_the_owner(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    superuser_token_headers: dict[str, str],
    normal_user_token_headers: dict[str, str],
) -> None:
    r = save(client, superuser_token_headers, video[0], None)
    assert r.status_code == 201
    owner = crud.get_user_by_email(session=db, email=settings.EMAIL_TEST_USER)
    assert owner and r.json()["owner_id"] == str(owner.id)
    assert (
        client.get(reel_url(video[0]), headers=normal_user_token_headers).status_code
        == 200
    )


def test_other_users_and_missing_videos_get_a_plain_404(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    save(client, normal_user_token_headers, video[0], None)
    email, password = random_email(), random_lower_string()
    crud.create_user(session=db, user_create=UserCreate(email=email, password=password))
    stranger = user_authentication_headers(
        client=client, email=email, password=password
    )
    for video_id, headers in [
        (video[0], stranger),
        (uuid.uuid4(), normal_user_token_headers),
    ]:
        assert client.get(reel_url(video_id), headers=headers).json() == {
            "detail": "Video not found"
        }
        assert save(client, headers, video_id, None).status_code == 404
    assert client.get(reel_url(video[0])).status_code == 401


def test_deleting_the_video_deletes_its_reel(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    reel = save(client, normal_user_token_headers, video[0], None).json()
    client.delete(f"{API}/videos/{video[0]}", headers=normal_user_token_headers)
    db.expire_all()
    assert db.get(Reel, uuid.UUID(reel["id"])) is None


# --- the database allows one live reel per video -----------------------------------


def test_second_live_reel_is_impossible(
    db: Session, video: tuple[uuid.UUID, list[int]]
) -> None:
    v = db.get(Video, video[0])
    assert v
    first = crud.create_reel(session=db, video=v, title="a", edit=ReelEdit())
    assert first
    # crud: no error, just "already has one"
    assert crud.create_reel(session=db, video=v, title="b", edit=ReelEdit()) is None
    # the index itself, bypassing crud
    db.add(Reel(video_id=v.id, owner_id=v.owner_id, title="c", edit={}))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    # deleted reels don't count
    crud.soft_delete_reel(session=db, reel_id=first.id)
    assert crud.create_reel(session=db, video=v, title="d", edit=ReelEdit())


# --- deprecated endpoints, until the frontend switches -----------------------------


def test_deprecated_post_is_get_or_create(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    """The old "make a highlight" button POSTs on every click: the first click
    creates the reel, later ones open the same reel (body ignored)."""
    h, url = normal_user_token_headers, f"{API}/videos/{video[0]}/reels"
    first = client.post(url, headers=h, json={"title": "First click"})
    assert first.status_code == 201
    again = client.post(
        url, headers=h, json={"title": "Second click", "edit": {"view": "full"}}
    )
    assert again.status_code == 200
    assert again.json() == first.json()


def test_old_routes_are_marked_deprecated() -> None:
    paths = app.openapi()["paths"]
    old = [
        ("get", "/videos/{id}/reels"),
        ("post", "/videos/{id}/reels"),
        ("get", "/reels/{reel_id}"),
        ("put", "/reels/{reel_id}"),
        ("delete", "/reels/{reel_id}"),
    ]
    for method, path in old:
        assert paths[API + path][method].get("deprecated") is True, (method, path)
    for method in ("get", "put"):
        assert not paths[API + "/videos/{id}/reel"][method].get("deprecated")
