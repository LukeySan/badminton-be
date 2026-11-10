import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, col, select

from app import crud
from app.core.config import settings
from app.models import CAPTION_MAX_LENGTH, Rally, Reel, UserCreate, Video
from tests.utils.user import user_authentication_headers
from tests.utils.utils import random_email, random_lower_string

API = settings.API_V1_STR
DURATION = 120.0
RALLIES = [(10.0, 20.0), (30.0, 45.0)]


# --- fixtures ---------------------------------------------------------------


def when(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp)


def make_video(db: Session, email: str) -> tuple[uuid.UUID, list[int]]:
    """A video row with two rallies (no file needed for reels)."""
    owner = crud.get_user_by_email(session=db, email=email)
    assert owner
    video = Video(
        owner_id=owner.id,
        filename="match.mp4",
        storage_path="x/original.mp4",
        status="done",
        duration=DURATION,
        fps=30.0,
    )
    db.add(video)
    db.commit()
    rallies = [Rally(video_id=video.id, start=s, end=e) for s, e in RALLIES]
    db.add_all(rallies)
    db.commit()
    return video.id, [r.id for r in rallies if r.id is not None]


@pytest.fixture
def video(db: Session) -> tuple[uuid.UUID, list[int]]:
    """The normal test user's video: (video id, rally ids)."""
    return make_video(db, settings.EMAIL_TEST_USER)


@pytest.fixture(scope="module")
def other_user_headers(client: TestClient, db: Session) -> dict[str, str]:
    email, password = random_email(), random_lower_string()
    crud.create_user(session=db, user_create=UserCreate(email=email, password=password))
    return user_authentication_headers(client=client, email=email, password=password)


def full_edit(rally_ids: list[int]) -> dict[str, Any]:
    return {
        "version": 1,
        "view": "full",
        "format": "9:16",
        "clips": [
            {"rally_id": rally_ids[1], "trim_start": 29.0, "trim_end": 46.0, "slow_mo": True},
            {"rally_id": rally_ids[0], "trim_start": 9.5, "trim_end": 20.5, "slow_mo": False},
        ],
        "restored_gaps": [{"start": 20.5, "end": 24.0}],
        "overlays": {
            "trail": True,
            "captions": [{"text": "What a smash", "start": 40.0, "end": 43.0}],
        },
    }  # fmt: skip


CreateFn = Callable[..., dict[str, Any]]


@pytest.fixture
def create(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> CreateFn:
    """create(title=..., edit=...) -> the new reel (asserts 201)."""

    def _create(title: str = "Best rallies", **body: Any) -> dict[str, Any]:
        r = client.post(
            f"{API}/videos/{video[0]}/reels",
            headers=normal_user_token_headers,
            json={"title": title, **body},
        )
        assert r.status_code == 201, r.text
        data: dict[str, Any] = r.json()
        return data

    return _create


# --- create / read ----------------------------------------------------------


def test_create_and_read_reel(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    create: CreateFn,
) -> None:
    video_id, rally_ids = video
    edit = full_edit(rally_ids)
    reel = create(edit=edit)
    assert reel["title"] == "Best rallies"
    assert reel["video_id"] == str(video_id)
    assert reel["edit"] == edit  # stored exactly, clips in reel order
    assert reel["created_at"] == reel["updated_at"]

    r = client.get(f"{API}/reels/{reel['id']}", headers=normal_user_token_headers)
    assert r.status_code == 200
    assert r.json() == reel


def test_new_reel_gets_an_empty_edit_with_defaults(create: CreateFn) -> None:
    reel = create()
    assert reel["edit"] == {
        "version": 1,
        "view": "cut",
        "format": "16:9",
        "clips": [],
        "restored_gaps": [],
        "overlays": {"trail": False, "captions": []},
    }


def test_reel_belongs_to_the_video_owner_even_when_a_superuser_makes_it(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    superuser_token_headers: dict[str, str],
    normal_user_token_headers: dict[str, str],
) -> None:
    r = client.post(
        f"{API}/videos/{video[0]}/reels",
        headers=superuser_token_headers,
        json={"title": "Made by support"},
    )
    assert r.status_code == 201
    owner = crud.get_user_by_email(session=db, email=settings.EMAIL_TEST_USER)
    assert owner
    assert r.json()["owner_id"] == str(owner.id)
    # ...so the owner sees and can open it
    r = client.get(f"{API}/reels/{r.json()['id']}", headers=normal_user_token_headers)
    assert r.status_code == 200


def test_list_reels_most_recently_saved_first(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    create: CreateFn,
) -> None:
    first, second, deleted = create("First"), create("Second"), create("Deleted")
    client.delete(f"{API}/reels/{deleted['id']}", headers=normal_user_token_headers)
    other_video, _ = make_video(db, settings.EMAIL_TEST_USER)
    client.post(
        f"{API}/videos/{other_video}/reels",
        headers=normal_user_token_headers,
        json={"title": "Other video's reel"},
    )
    # Saving `first` again moves it to the top
    r = client.put(
        f"{API}/reels/{first['id']}",
        headers=normal_user_token_headers,
        json={
            "title": "First",
            "edit": first["edit"],
            "updated_at": first["updated_at"],
        },
    )
    assert r.status_code == 200

    r = client.get(f"{API}/videos/{video[0]}/reels", headers=normal_user_token_headers)
    assert r.status_code == 200
    assert [reel["title"] for reel in r.json()["data"]] == ["First", "Second"]
    assert r.json()["count"] == 2

    r = client.get(
        f"{API}/videos/{video[0]}/reels?limit=1", headers=normal_user_token_headers
    )
    assert [reel["id"] for reel in r.json()["data"]] == [first["id"]]
    assert r.json()["count"] == 2  # the total, not the page size
    assert second["id"] != first["id"]


# --- update -----------------------------------------------------------------


def test_update_replaces_title_and_whole_edit(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    create: CreateFn,
) -> None:
    reel = create(edit=full_edit(video[1]))
    new_edit = {
        "view": "cut",
        "clips": [{"rally_id": video[1][0], "trim_start": 10, "trim_end": 19}],
    }
    r = client.put(
        f"{API}/reels/{reel['id']}",
        headers=normal_user_token_headers,
        json={"title": "Renamed", "edit": new_edit, "updated_at": reel["updated_at"]},
    )  # fmt: skip
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["title"] == "Renamed"
    # Full replace: what wasn't sent is back to the defaults, not merged
    assert saved["edit"]["format"] == "16:9"
    assert saved["edit"]["restored_gaps"] == []
    assert saved["edit"]["overlays"] == {"trail": False, "captions": []}
    assert saved["edit"]["clips"] == [
        {
            "rally_id": video[1][0],
            "trim_start": 10.0,
            "trim_end": 19.0,
            "slow_mo": False,
        }
    ]
    assert when(saved["updated_at"]) > when(reel["updated_at"])
    assert saved["created_at"] == reel["created_at"]


def test_stale_save_is_rejected(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    create: CreateFn,
) -> None:
    """Two tabs open the same reel; the second one to save gets a 409 instead of
    silently overwriting the first."""
    reel = create()
    loaded_at = reel["updated_at"]
    url = f"{API}/reels/{reel['id']}"
    tab_a = client.put(
        url,
        headers=normal_user_token_headers,
        json={"title": "From tab A", "edit": {}, "updated_at": loaded_at},
    )
    assert tab_a.status_code == 200
    tab_b = client.put(
        url,
        headers=normal_user_token_headers,
        json={"title": "From tab B", "edit": {}, "updated_at": loaded_at},
    )
    assert tab_b.status_code == 409
    assert tab_b.json()["code"] == "reel_changed"
    assert "reload" in tab_b.json()["detail"]
    assert (
        client.get(url, headers=normal_user_token_headers).json()["title"]
        == "From tab A"
    )
    # With the fresh updated_at, tab B can save
    r = client.put(
        url,
        headers=normal_user_token_headers,
        json={
            "title": "From tab B",
            "edit": {},
            "updated_at": tab_a.json()["updated_at"],
        },
    )
    assert r.status_code == 200


def test_update_needs_a_timezone_aware_updated_at(
    client: TestClient, normal_user_token_headers: dict[str, str], create: CreateFn
) -> None:
    reel = create()
    naive = reel["updated_at"].replace("Z", "").split("+")[0]
    r = client.put(
        f"{API}/reels/{reel['id']}",
        headers=normal_user_token_headers,
        json={"title": "x", "edit": {}, "updated_at": naive},
    )
    assert r.status_code == 422


def test_update_requires_updated_at(
    client: TestClient, normal_user_token_headers: dict[str, str], create: CreateFn
) -> None:
    reel = create()
    r = client.put(
        f"{API}/reels/{reel['id']}",
        headers=normal_user_token_headers,
        json={"title": "x", "edit": {}},
    )
    assert r.status_code == 422


# --- delete (soft) ----------------------------------------------------------


def test_delete_is_soft(
    client: TestClient,
    db: Session,
    normal_user_token_headers: dict[str, str],
    create: CreateFn,
) -> None:
    reel = create()
    url = f"{API}/reels/{reel['id']}"
    r = client.delete(url, headers=normal_user_token_headers)
    assert r.status_code == 200
    assert r.json() == {"message": "Reel deleted successfully"}

    # Gone from the API...
    assert client.get(url, headers=normal_user_token_headers).status_code == 404
    r = client.put(
        url,
        headers=normal_user_token_headers,
        json={"title": "x", "edit": {}, "updated_at": reel["updated_at"]},
    )
    assert r.status_code == 404
    assert client.delete(url, headers=normal_user_token_headers).status_code == 404
    # ...but the row is still there
    db.expire_all()
    row = db.get(Reel, uuid.UUID(reel["id"]))
    assert row and row.deleted_at is not None


def test_deleting_the_video_deletes_its_reels(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    create: CreateFn,
) -> None:
    reel = create()
    r = client.delete(f"{API}/videos/{video[0]}", headers=normal_user_token_headers)
    assert r.status_code == 200
    db.expire_all()
    assert db.get(Reel, uuid.UUID(reel["id"])) is None


# --- access -----------------------------------------------------------------


def test_other_users_get_404_everywhere(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    other_user_headers: dict[str, str],
    create: CreateFn,
) -> None:
    reel = create()
    h = other_user_headers
    reel_url, video_url = f"{API}/reels/{reel['id']}", f"{API}/videos/{video[0]}/reels"
    body = {"title": "x", "edit": {}, "updated_at": reel["updated_at"]}
    assert client.get(reel_url, headers=h).status_code == 404
    assert client.put(reel_url, headers=h, json=body).status_code == 404
    assert client.delete(reel_url, headers=h).status_code == 404
    assert client.get(video_url, headers=h).status_code == 404
    assert client.post(video_url, headers=h, json={"title": "x"}).status_code == 404


def test_superuser_can_open_and_save_any_reel(
    client: TestClient, superuser_token_headers: dict[str, str], create: CreateFn
) -> None:
    reel = create()
    url = f"{API}/reels/{reel['id']}"
    assert client.get(url, headers=superuser_token_headers).status_code == 200
    r = client.put(
        url,
        headers=superuser_token_headers,
        json={"title": "Fixed", "edit": {}, "updated_at": reel["updated_at"]},
    )
    assert r.status_code == 200


def test_reels_need_login(
    client: TestClient, video: tuple[uuid.UUID, list[int]], create: CreateFn
) -> None:
    reel = create()
    assert client.get(f"{API}/reels/{reel['id']}").status_code == 401
    assert client.get(f"{API}/videos/{video[0]}/reels").status_code == 401


def test_missing_reel_and_video(
    client: TestClient, normal_user_token_headers: dict[str, str]
) -> None:
    h = normal_user_token_headers
    assert client.get(f"{API}/reels/{uuid.uuid4()}", headers=h).status_code == 404
    r = client.post(
        f"{API}/videos/{uuid.uuid4()}/reels", headers=h, json={"title": "x"}
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "Video not found"


# --- validation -------------------------------------------------------------


def clip(
    rally_id: int | None, start: float = 10.0, end: float = 20.0
) -> dict[str, Any]:
    return {"rally_id": rally_id, "trim_start": start, "trim_end": end}


@pytest.mark.parametrize(
    ("edit", "loc"),
    [
        # Checked against the video
        pytest.param({"clips": [clip(999_999_999)]}, ["clips", 0, "rally_id"], id="unknown-rally"),
        pytest.param({"clips": [clip(None, 100, DURATION + 1)]}, ["clips", 0, "trim_end"], id="trim-past-end"),
        pytest.param({"restored_gaps": [{"start": 110, "end": DURATION + 1}]}, ["restored_gaps", 0, "end"], id="gap-past-end"),
        pytest.param({"overlays": {"captions": [{"text": "x", "start": 110, "end": DURATION + 1}]}}, ["overlays", "captions", 0, "end"], id="caption-past-end"),
        # Pydantic
        pytest.param({"clips": [clip(None, 20, 20)]}, ["clips", 0], id="empty-trim"),
        pytest.param({"clips": [clip(None, 20, 10)]}, ["clips", 0], id="trim-backwards"),
        pytest.param({"clips": [clip(None, -1, 10)]}, ["clips", 0, "trim_start"], id="negative-trim"),
        pytest.param({"clips": [{"trim_start": 1, "trim_end": 2}]}, ["clips", 0, "rally_id"], id="rally-id-required"),
        pytest.param({"restored_gaps": [{"start": 5, "end": 4}]}, ["restored_gaps", 0], id="gap-backwards"),
        pytest.param({"overlays": {"captions": [{"text": "x" * (CAPTION_MAX_LENGTH + 1), "start": 1, "end": 2}]}}, ["overlays", "captions", 0, "text"], id="caption-81-chars"),
        pytest.param({"overlays": {"captions": [{"text": "", "start": 1, "end": 2}]}}, ["overlays", "captions", 0, "text"], id="caption-empty"),
        pytest.param({"view": "split"}, ["view"], id="bad-view"),
        pytest.param({"format": "1:1"}, ["format"], id="bad-format"),
        pytest.param({"version": 2}, ["version"], id="unknown-version"),
        pytest.param({"clipz": []}, ["clipz"], id="unknown-field"),
        pytest.param({"clips": [{**clip(None), "speed": 0.5}]}, ["clips", 0, "speed"], id="unknown-clip-field"),
    ],
)  # fmt: skip
def test_invalid_edit_is_rejected(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    edit: dict[str, Any],
    loc: list[str | int],
) -> None:
    r = client.post(
        f"{API}/videos/{video[0]}/reels",
        headers=normal_user_token_headers,
        json={"title": "x", "edit": edit},
    )
    assert r.status_code == 422, r.text
    locs = [e["loc"] for e in r.json()["detail"]]
    assert ["body", "edit", *loc] in locs, locs


def test_rally_of_another_video_is_rejected(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    _, other_rallies = make_video(db, settings.EMAIL_TEST_USER)
    r = client.post(
        f"{API}/videos/{video[0]}/reels",
        headers=normal_user_token_headers,
        json={"title": "x", "edit": {"clips": [clip(other_rallies[0])]}},
    )
    assert r.status_code == 422
    assert r.json()["detail"][0]["msg"] == "not a rally of this video"


def test_all_problems_are_reported_at_once(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    edit = {
        "clips": [clip(video[1][0]), clip(999_999_999), clip(None, 100, 500)],
        "overlays": {"captions": [{"text": "x", "start": 1, "end": 500}]},
    }
    r = client.post(
        f"{API}/videos/{video[0]}/reels",
        headers=normal_user_token_headers,
        json={"title": "x", "edit": edit},
    )
    assert r.status_code == 422
    assert [e["loc"] for e in r.json()["detail"]] == [
        ["body", "edit", "clips", 1, "rally_id"],
        ["body", "edit", "clips", 2, "trim_end"],
        ["body", "edit", "overlays", "captions", 0, "end"],
    ]
    assert "within the video (0-120.00 s)" in r.json()["detail"][1]["msg"]


def test_limits_that_are_allowed(
    create: CreateFn, video: tuple[uuid.UUID, list[int]]
) -> None:
    """Edges: an 80-character caption, a trim ending exactly at (or a hair past)
    the end of the video, a clip not tied to any rally, a clip used twice."""
    edit = {
        "clips": [
            clip(None, 100, DURATION + 0.05),  # player's duration can differ a bit
            clip(video[1][0]),
            {**clip(video[1][0]), "slow_mo": True},
        ],
        "restored_gaps": [{"start": 0, "end": DURATION}],
        "overlays": {"captions": [{"text": "x" * CAPTION_MAX_LENGTH, "start": 0, "end": DURATION}]},
    }  # fmt: skip
    reel = create(edit=edit)
    assert len(reel["edit"]["clips"]) == 3


@pytest.mark.parametrize("title", ["", "x" * 101])
def test_title_length(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    title: str,
) -> None:
    r = client.post(
        f"{API}/videos/{video[0]}/reels",
        headers=normal_user_token_headers,
        json={"title": title},
    )
    assert r.status_code == 422


def test_update_is_validated_against_the_video_too(
    client: TestClient,
    normal_user_token_headers: dict[str, str],
    create: CreateFn,
) -> None:
    reel = create()
    r = client.put(
        f"{API}/reels/{reel['id']}",
        headers=normal_user_token_headers,
        json={
            "title": "x",
            "edit": {"clips": [clip(999_999_999)]},
            "updated_at": reel["updated_at"],
        },
    )
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "edit", "clips", 0, "rally_id"]
    # Nothing was saved, so the original updated_at still works
    r = client.put(
        f"{API}/reels/{reel['id']}",
        headers=normal_user_token_headers,
        json={"title": "x", "edit": {}, "updated_at": reel["updated_at"]},
    )
    assert r.status_code == 200


def test_reel_rows_store_plain_json(
    db: Session, create: CreateFn, video: tuple[uuid.UUID, list[int]]
) -> None:
    """The edit list is a JSONB document with the version inside."""
    create(edit=full_edit(video[1]))
    row = db.exec(select(Reel).where(col(Reel.video_id) == video[0])).one()
    assert row.edit["version"] == 1
    assert row.edit["clips"][0]["slow_mo"] is True
