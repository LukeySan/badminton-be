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
    """Rallies are 10-20 s and 30-45 s; pieces may go 1 s beyond (default lead)."""
    return {
        "version": 2,
        "view": "full",
        "format": "9:16",
        "lead_in": 1.0,
        "lead_out": 2.5,
        "clips": [
            {"rally_id": rally_ids[1], "pieces": [
                {"start": 29.0, "end": 35.0, "slow_mo": False},
                {"start": 35.0, "end": 38.0, "slow_mo": True},  # touching: a split
                {"start": 41.0, "end": 47.5, "slow_mo": False},
            ]},
            {"rally_id": rally_ids[0], "pieces": [
                {"start": 9.5, "end": 20.5, "slow_mo": False},
            ]},
        ],
        "restored_gaps": [{"start": 20.5, "end": 24.0}],
        "overlays": {
            "trail": True,
            "captions": [{"text": "What a smash", "start": 40.0, "end": 43.0}],
        },
    }  # fmt: skip


def piece(start: float, end: float, slow_mo: bool = False) -> dict[str, Any]:
    return {"start": start, "end": end, "slow_mo": slow_mo}


def clip(rally_id: int, *pieces: dict[str, Any]) -> dict[str, Any]:
    return {"rally_id": rally_id, "pieces": list(pieces)}


CreateFn = Callable[..., dict[str, Any]]


@pytest.fixture
def create(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> CreateFn:
    """create(title=..., edit=...) -> the video's new reel, made with
    PUT /videos/{id}/reel (asserts 201)."""

    def _create(
        title: str = "Best rallies", edit: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        r = client.put(
            f"{API}/videos/{video[0]}/reel",
            headers=normal_user_token_headers,
            json={"title": title, "edit": edit or {}, "updated_at": None},
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
        "version": 2,
        "view": "cut",
        "format": "16:9",
        "lead_in": 1.0,
        "lead_out": 1.0,
        "clips": [],
        "restored_gaps": [],
        "overlays": {"trail": False, "captions": []},
    }


def test_clips_without_pieces_are_dropped(
    create: CreateFn, video: tuple[uuid.UUID, list[int]]
) -> None:
    """Deleting a clip's last piece in the editor leaves an empty clip: the save
    drops it, so stored reels never have empty clips."""
    a, b = video[1]
    reel = create(edit={"clips": [clip(b), clip(a, piece(10, 12)), clip(b)]})
    assert reel["edit"]["clips"] == [clip(a, piece(10, 12))]


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


def test_deprecated_list_has_the_one_live_reel(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    create: CreateFn,
) -> None:
    h = normal_user_token_headers
    first = create("First")
    client.delete(f"{API}/reels/{first['id']}", headers=h)
    second = create("Second")  # possible again once the first is deleted
    other_video, _ = make_video(db, settings.EMAIL_TEST_USER)
    client.put(
        f"{API}/videos/{other_video}/reel",
        headers=h,
        json={"title": "Other video's reel", "edit": {}},
    )
    r = client.get(f"{API}/videos/{video[0]}/reels", headers=h)
    assert r.status_code == 200
    assert [reel["id"] for reel in r.json()["data"]] == [second["id"]]
    assert r.json()["count"] == 1


# --- update -----------------------------------------------------------------


def test_update_replaces_title_and_whole_edit(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    create: CreateFn,
) -> None:
    reel = create(edit=full_edit(video[1]))
    new_edit = {"view": "cut", "clips": [clip(video[1][0], piece(10, 19))]}
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
    assert saved["edit"]["lead_out"] == 1.0
    assert saved["edit"]["restored_gaps"] == []
    assert saved["edit"]["overlays"] == {"trail": False, "captions": []}
    assert saved["edit"]["clips"] == [clip(video[1][0], piece(10, 19))]
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
# The video is 120 s; rally A is 10-20 s, rally B 30-45 s; default lead 1 s.

A, B = 0, 1  # index into the video's rally ids


@pytest.mark.parametrize(
    ("edit", "loc"),
    [
        # Checked against the video
        pytest.param({"clips": [{"rally_id": 999_999_999, "pieces": [piece(10, 12)]}]}, ["clips", 0, "rally_id"], id="unknown-rally"),
        pytest.param({"clips": [[A, piece(8.9, 12)]]}, ["clips", 0, "pieces", 0, "start"], id="before-lead-in"),
        pytest.param({"clips": [[A, piece(12, 21.1)]]}, ["clips", 0, "pieces", 0, "end"], id="after-lead-out"),
        pytest.param({"lead_out": 3, "clips": [[A, piece(12, 23.1)]]}, ["clips", 0, "pieces", 0, "end"], id="after-max-lead-out"),
        pytest.param({"clips": [[A, piece(10, 11), piece(30, 31)]]}, ["clips", 0, "pieces", 1, "end"], id="piece-in-another-rally"),
        pytest.param({"restored_gaps": [{"start": 110, "end": DURATION + 1}]}, ["restored_gaps", 0, "end"], id="gap-past-end"),
        pytest.param({"overlays": {"captions": [{"text": "x", "start": 110, "end": DURATION + 1}]}}, ["overlays", "captions", 0, "end"], id="caption-past-end"),
        # Pydantic
        pytest.param({"clips": [[A, piece(12, 12)]]}, ["clips", 0, "pieces", 0], id="empty-piece"),
        pytest.param({"clips": [[A, piece(14, 12)]]}, ["clips", 0, "pieces", 0], id="piece-backwards"),
        pytest.param({"clips": [[A, piece(14, 16), piece(11, 12)]]}, ["clips", 0], id="pieces-unsorted"),
        pytest.param({"clips": [[A, piece(11, 15), piece(14, 16)]]}, ["clips", 0], id="pieces-overlap"),
        pytest.param({"clips": [{"pieces": []}]}, ["clips", 0, "rally_id"], id="rally-id-required"),
        pytest.param({"clips": [{"rally_id": None, "pieces": []}]}, ["clips", 0, "rally_id"], id="rally-id-null"),
        pytest.param({"lead_in": 3.5}, ["lead_in"], id="lead-in-over-3"),
        pytest.param({"lead_out": -1}, ["lead_out"], id="negative-lead-out"),
        pytest.param({"restored_gaps": [{"start": 5, "end": 4}]}, ["restored_gaps", 0], id="gap-backwards"),
        pytest.param({"overlays": {"captions": [{"text": "x" * (CAPTION_MAX_LENGTH + 1), "start": 1, "end": 2}]}}, ["overlays", "captions", 0, "text"], id="caption-81-chars"),
        pytest.param({"overlays": {"captions": [{"text": "", "start": 1, "end": 2}]}}, ["overlays", "captions", 0, "text"], id="caption-empty"),
        pytest.param({"view": "split"}, ["view"], id="bad-view"),
        pytest.param({"format": "1:1"}, ["format"], id="bad-format"),
        pytest.param({"version": 1}, ["version"], id="old-version"),
        pytest.param({"clips": [{"rally_id": 1, "trim_start": 1, "trim_end": 2}]}, ["clips", 0, "trim_start"], id="v1-clip"),
        pytest.param({"clipz": []}, ["clipz"], id="unknown-field"),
        pytest.param({"clips": [[A, {**piece(11, 12), "speed": 0.5}]]}, ["clips", 0, "pieces", 0, "speed"], id="unknown-piece-field"),
    ],
)  # fmt: skip
def test_invalid_edit_is_rejected(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    edit: dict[str, Any],
    loc: list[str | int],
) -> None:
    # [A, piece...] is shorthand for a clip of rally A
    clips = [
        clip(video[1][c[0]], *c[1:]) if isinstance(c, list) else c
        for c in edit.get("clips", [])
    ]
    body = {**edit, "clips": clips} if "clips" in edit else edit
    r = client.put(
        f"{API}/videos/{video[0]}/reel",
        headers=normal_user_token_headers,
        json={"title": "x", "edit": body},
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
    r = client.put(
        f"{API}/videos/{video[0]}/reel",
        headers=normal_user_token_headers,
        json={"title": "x", "edit": {"clips": [clip(other_rallies[0], piece(10, 12))]}},
    )
    assert r.status_code == 422
    assert r.json()["detail"][0]["msg"] == "not a rally of this video"


def test_all_problems_are_reported_at_once(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    a, b = video[1]
    edit = {
        "clips": [
            clip(a, piece(10, 12)),
            clip(999_999_999, piece(1, 2)),
            clip(b, piece(20, 31), piece(44, 50)),
        ],
        "overlays": {"captions": [{"text": "x", "start": 1, "end": 500}]},
    }
    r = client.put(
        f"{API}/videos/{video[0]}/reel",
        headers=normal_user_token_headers,
        json={"title": "x", "edit": edit},
    )
    assert r.status_code == 422
    assert [e["loc"] for e in r.json()["detail"]] == [
        ["body", "edit", "clips", 1, "rally_id"],
        ["body", "edit", "clips", 2, "pieces", 0, "start"],
        ["body", "edit", "clips", 2, "pieces", 1, "end"],
        ["body", "edit", "overlays", "captions", 0, "end"],
    ]
    assert r.json()["detail"][1]["msg"] == (
        "must be within the rally (30.00-45.00 s) widened by lead_in/lead_out "
        "(29.00-46.00 s)"
    )
    assert "within the video (0-120.00 s)" in r.json()["detail"][3]["msg"]


def test_limits_that_are_allowed(
    client: TestClient,
    db: Session,
    normal_user_token_headers: dict[str, str],
    create: CreateFn,
    video: tuple[uuid.UUID, list[int]],
) -> None:
    """Edges: pieces exactly at the padded bounds (and a hair past, as JavaScript
    may compute them), touching pieces, the maximum 3 s lead, an 80-character
    caption, a rally used by two clips."""
    a, b = video[1]
    edit = {
        "lead_in": 3,
        "lead_out": 3,
        "clips": [
            clip(a, piece(7.0 - 0.005, 15), piece(15, 23.0 + 0.005, slow_mo=True)),
            clip(b, piece(27, 48)),
            clip(a, piece(10, 20)),
        ],
        "restored_gaps": [{"start": 0, "end": DURATION}],
        "overlays": {"captions": [{"text": "x" * CAPTION_MAX_LENGTH, "start": 0, "end": DURATION}]},
    }  # fmt: skip
    reel = create(edit=edit)
    assert len(reel["edit"]["clips"]) == 3
    # A rally near the end of the video: past its 3 s lead-out, the end of the
    # video is the limit (with the 0.1 s player tolerance)
    late_video, _ = make_video(db, settings.EMAIL_TEST_USER)
    late = Rally(video_id=late_video, start=110.0, end=119.0)
    db.add(late)
    db.commit()
    for end, status in [(DURATION + 0.05, 201), (DURATION + 0.5, 422)]:
        late_clip = clip(late.id or 0, piece(112, end))
        r = client.put(
            f"{API}/videos/{late_video}/reel",
            headers=normal_user_token_headers,
            json={"title": "x", "edit": {"lead_out": 3, "clips": [late_clip]}},
        )
        assert r.status_code == status, r.text
    assert "within the video" in r.json()["detail"][0]["msg"]


@pytest.mark.parametrize("title", ["", "x" * 101])
def test_title_length(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    title: str,
) -> None:
    r = client.put(
        f"{API}/videos/{video[0]}/reel",
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
            "edit": {"clips": [clip(999_999_999, piece(1, 2))]},
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
    assert row.edit["version"] == 2
    assert row.edit["clips"][0]["pieces"][1]["slow_mo"] is True
