import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, col, select

from app import crud
from app.core.config import settings
from app.models import (
    MAX_OVERLAYS,
    MAX_REEL_SEGMENTS,
    OVERLAY_ID_MAX_LENGTH,
    OVERLAY_SIZE_MAX,
    OVERLAY_SIZE_MIN,
    OVERLAY_TEXT_MAX_LENGTH,
    SEGMENT_ID_MAX_LENGTH,
    Rally,
    Reel,
    ReelEdit,
    UserCreate,
    Video,
)
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
    """Rallies are 10-20 s and 30-45 s. Segments in playback order: rally 2, then
    rally 1 in two parts (the second slow), a replay of part of rally 2, and footage
    from outside the rallies (45.5 s of reel); a text overlay over the replay."""
    a, b = rally_ids
    return {
        "version": 6,
        "format": "9:16",
        "segments": [
            seg("b", 29.0, 46.0, rally_id=b),
            seg("a1", 9.0, 15.0, rally_id=a),
            seg("a2", 15.0, 21.0, slow_mo=True, rally_id=a),
            seg("b-replay", 40.0, 43.0, slow_mo=True, rally_id=b),
            seg("crowd", 50.0, 54.5),
        ],
        "overlays": [overlay("smash", 35.0, 41.0, text="What a smash", y=0.2)],
    }


def seg(
    id: str,
    start: float,
    end: float,
    slow_mo: bool = False,
    rally_id: int | None = None,
) -> dict[str, Any]:
    return {"id": id, "start": start, "end": end, "slow_mo": slow_mo,
            "rally_id": rally_id}  # fmt: skip


def overlay(id: str, start: float, end: float, **kw: Any) -> dict[str, Any]:
    """A text overlay as stored (all fields, defaults filled in)."""
    return {"id": id, "type": "text", "text": kw.pop("text", "Nice"), "start": start,
            "end": end, "x": kw.pop("x", 0.5), "y": kw.pop("y", 0.85),
            "size": kw.pop("size", 0.06), **kw}  # fmt: skip


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
    assert reel["edit"] == edit  # stored exactly, in playback order
    assert reel["created_at"] == reel["updated_at"]

    r = client.get(f"{API}/reels/{reel['id']}", headers=normal_user_token_headers)
    assert r.status_code == 200
    assert r.json() == reel


def test_new_reel_gets_an_empty_edit_with_defaults(create: CreateFn) -> None:
    reel = create()
    assert reel["edit"] == {
        "version": 6,
        "format": "16:9",
        "segments": [],
        "overlays": [],
    }


def test_times_are_stored_to_tenths(create: CreateFn) -> None:
    """0.1 s precision: the app may send JavaScript's 12.300000000000001."""
    reel = create(edit={"segments": [seg("a", 12.300000000000001, 14.04)],
                        "overlays": [overlay("o", 0.3000000000000001, 1.04)]})  # fmt: skip
    assert reel["edit"]["segments"] == [seg("a", 12.3, 14.0)]
    assert reel["edit"]["overlays"] == [overlay("o", 0.3, 1.0)]


def test_overlays_are_clamped_to_the_reel_on_save(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    create: CreateFn,
) -> None:
    """Overlays are in reel time, so they end with the reel: one running past the
    end is cut there, one starting too late keeps the last 0.3 s. When a later save
    shrinks the reel, its overlays are clamped again. A reel without segments has no
    length yet: its overlays are kept as they are."""
    h = normal_user_token_headers
    url = f"{API}/videos/{video[0]}/reel"

    def spans(edit: dict[str, Any]) -> list[tuple[str, float, float]]:
        return [(o["id"], o["start"], o["end"]) for o in edit["overlays"]]

    # 5 s + 2.5 s of slow-mo (5 s) = a 10 s reel
    reel = create(edit={
        "segments": [seg("a", 10, 15), seg("b", 30, 32.5, slow_mo=True)],
        "overlays": [overlay("inside", 1, 4), overlay("across", 8, 12),
                     overlay("past", 11, 14), overlay("last", 9.7, 10)],
    })  # fmt: skip
    assert spans(reel["edit"]) == [
        ("inside", 1, 4),
        ("across", 8, 10),
        ("past", 9.7, 10),
        ("last", 9.7, 10),
    ]
    assert client.get(url, headers=h).json()["edit"] == reel["edit"]  # as stored

    # The reel shrinks to 3 s
    edit = {**reel["edit"], "segments": [seg("a", 10, 13)]}
    r = client.put(url, headers=h, json={"title": "x", "edit": edit,
                                         "updated_at": reel["updated_at"]})  # fmt: skip
    assert r.status_code == 200, r.text
    reel = r.json()
    assert spans(reel["edit"]) == [
        ("inside", 1, 3),
        ("across", 2.7, 3),
        ("past", 2.7, 3),
        ("last", 2.7, 3),
    ]
    # Everything else about them is kept
    assert reel["edit"]["overlays"][0] == overlay("inside", 1, 3)

    # No segments: nothing to clamp to
    edit = {**reel["edit"], "segments": []}
    r = client.put(url, headers=h, json={"title": "x", "edit": edit,
                                         "updated_at": reel["updated_at"]})  # fmt: skip
    assert r.status_code == 200, r.text
    assert r.json()["edit"]["overlays"] == reel["edit"]["overlays"]


def test_reel_length_counts_slow_mo_double() -> None:
    edit = ReelEdit.model_validate(
        {"segments": [seg("a", 1, 2.3), seg("b", 5, 5.7, slow_mo=True), seg("c", 1, 2)]}
    )
    assert edit.length == 1.3 + 1.4 + 1.0
    assert ReelEdit().length == 0


def test_labels_of_other_videos_rallies_are_stored_as_null(
    db: Session, create: CreateFn, video: tuple[uuid.UUID, list[int]]
) -> None:
    """rally_id is a label, never a reason to refuse a save."""
    _, other_rallies = make_video(db, settings.EMAIL_TEST_USER)
    a, _ = video[1]
    reel = create(edit={"segments": [seg("x", 1, 2, rally_id=other_rallies[0]),
                                     seg("y", 10, 12, rally_id=a)]})  # fmt: skip
    assert [s["rally_id"] for s in reel["edit"]["segments"]] == [None, a]


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
    new_edit = {"segments": [seg("only", 10, 19, rally_id=video[1][0])]}
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
    assert saved["edit"]["overlays"] == []
    assert saved["edit"]["segments"] == [seg("only", 10, 19, rally_id=video[1][0])]
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
# The video is 120 s; rally A is 10-20 s, rally B 30-45 s.


def many(n: int, length: float = 0.5) -> list[dict[str, Any]]:
    return [seg(f"s{i}", 1.0, 1.0 + length) for i in range(n)]


@pytest.mark.parametrize(
    ("edit", "loc"),
    [
        # Checked against the video
        pytest.param({"segments": [seg("a", 110, DURATION + 0.2)]}, ["segments", 0, "end"], id="segment-past-end"),
        # Pydantic
        pytest.param({"segments": [seg("a", 12, 12.2)]}, ["segments", 0], id="shorter-than-0.3"),
        pytest.param({"segments": [seg("a", 12, 12.24)]}, ["segments", 0], id="0.24-rounds-to-0.2"),
        pytest.param({"segments": [seg("a", 14, 12)]}, ["segments", 0], id="backwards"),
        pytest.param({"segments": [seg("a", -1, 2)]}, ["segments", 0, "start"], id="negative-start"),
        pytest.param({"segments": [seg("a", 1, 2), seg("a", 3, 4)]}, ["segments"], id="repeated-id"),
        pytest.param({"segments": [seg("", 1, 2)]}, ["segments", 0, "id"], id="empty-id"),
        pytest.param({"segments": [seg("x" * (SEGMENT_ID_MAX_LENGTH + 1), 1, 2)]}, ["segments", 0, "id"], id="id-too-long"),
        pytest.param({"segments": [{"start": 1, "end": 2}]}, ["segments", 0, "id"], id="id-required"),
        pytest.param({"segments": many(MAX_REEL_SEGMENTS + 1)}, ["segments"], id="over-200-segments"),
        pytest.param({"segments": [{**seg("a", 1, 2), "speed": 0.5}]}, ["segments", 0, "speed"], id="unknown-segment-field"),
        # Overlays
        pytest.param({"overlays": [overlay("o", 1, 2, text="x" * (OVERLAY_TEXT_MAX_LENGTH + 1))]}, ["overlays", 0, "text", "text"], id="text-81-chars"),
        pytest.param({"overlays": [overlay("o", 1, 2, text="")]}, ["overlays", 0, "text", "text"], id="text-empty"),
        pytest.param({"overlays": [overlay("o", 3, 2)]}, ["overlays", 0, "text"], id="overlay-backwards"),
        pytest.param({"overlays": [overlay("o", 1, 1.2)]}, ["overlays", 0, "text"], id="overlay-shorter-than-0.3"),
        pytest.param({"overlays": [overlay("o", 1, 1.24)]}, ["overlays", 0, "text"], id="overlay-0.24-rounds-to-0.2"),
        pytest.param({"overlays": [overlay("o", -1, 2)]}, ["overlays", 0, "text", "start"], id="overlay-negative-start"),
        pytest.param({"overlays": [overlay("o", 1, 2, size=OVERLAY_SIZE_MIN - 0.001)]}, ["overlays", 0, "text", "size"], id="size-too-small"),
        pytest.param({"overlays": [overlay("o", 1, 2, size=OVERLAY_SIZE_MAX + 0.001)]}, ["overlays", 0, "text", "size"], id="size-too-big"),
        pytest.param({"overlays": [overlay("o", 1, 2, x=-0.01)]}, ["overlays", 0, "text", "x"], id="x-below-0"),
        pytest.param({"overlays": [overlay("o", 1, 2, x=1.01)]}, ["overlays", 0, "text", "x"], id="x-above-1"),
        pytest.param({"overlays": [overlay("o", 1, 2, y=-0.01)]}, ["overlays", 0, "text", "y"], id="y-below-0"),
        pytest.param({"overlays": [overlay("o", 1, 2, y=1.01)]}, ["overlays", 0, "text", "y"], id="y-above-1"),
        pytest.param({"overlays": [{k: v for k, v in overlay("o", 1, 2).items() if k != "x"}]}, ["overlays", 0, "text", "x"], id="x-required"),
        pytest.param({"overlays": [overlay("o", 1, 2), overlay("o", 3, 4)]}, ["overlays"], id="repeated-overlay-id"),
        pytest.param({"overlays": [overlay("", 1, 2)]}, ["overlays", 0, "text", "id"], id="empty-overlay-id"),
        pytest.param({"overlays": [overlay("x" * (OVERLAY_ID_MAX_LENGTH + 1), 1, 2)]}, ["overlays", 0, "text", "id"], id="overlay-id-too-long"),
        pytest.param({"overlays": [overlay(f"o{i}", 1, 2) for i in range(MAX_OVERLAYS + 1)]}, ["overlays"], id="over-20-overlays"),
        pytest.param({"overlays": [{**overlay("o", 1, 2), "color": "red"}]}, ["overlays", 0, "text", "color"], id="unknown-overlay-field"),
        # "score" is reserved for later, not accepted yet
        pytest.param({"overlays": [{**overlay("o", 1, 2), "type": "score"}]}, ["overlays", 0], id="score-not-yet"),
        pytest.param({"overlays": [{k: v for k, v in overlay("o", 1, 2).items() if k != "type"}]}, ["overlays", 0], id="type-required"),
        pytest.param({"captions": [{"text": "x", "start": 1, "end": 2}]}, ["captions"], id="v5-captions"),
        pytest.param({"format": "1:1"}, ["format"], id="bad-format"),
        pytest.param({"version": 5}, ["version"], id="v5"),
        pytest.param({"version": 4}, ["version"], id="v4"),
        pytest.param({"version": 1}, ["version"], id="old-version"),
        pytest.param({"clips": []}, ["clips"], id="v4-clips"),
        pytest.param({"lead_in": 1}, ["lead_in"], id="v4-lead"),
        pytest.param({"restored_anchors": [25]}, ["restored_anchors"], id="v4-anchors"),
        pytest.param({"overlays": {"captions": []}}, ["overlays"], id="v4-overlays"),
        pytest.param({"view": "cut"}, ["view"], id="v4-view"),
    ],
)  # fmt: skip
def test_invalid_edit_is_rejected(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    edit: dict[str, Any],
    loc: list[str | int],
) -> None:
    r = client.put(
        f"{API}/videos/{video[0]}/reel",
        headers=normal_user_token_headers,
        json={"title": "x", "edit": edit},
    )
    assert r.status_code == 422, r.text
    locs = [e["loc"] for e in r.json()["detail"]]
    assert ["body", "edit", *loc] in locs, locs


def test_without_a_duration_only_pydantic_checks_apply() -> None:
    """A video whose duration couldn't be read (never the case after upload)."""
    edit = ReelEdit.model_validate({"segments": [seg("a", 500, 600)]})
    assert edit.problems(duration=None) == []


def test_all_problems_are_reported_at_once(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    edit = {
        "segments": [seg("a", 10, 12), seg("b", 115, 121), seg("c", 100, 125)],
    }
    r = client.put(
        f"{API}/videos/{video[0]}/reel",
        headers=normal_user_token_headers,
        json={"title": "x", "edit": edit},
    )
    assert r.status_code == 422
    assert [e["loc"] for e in r.json()["detail"]] == [
        ["body", "edit", "segments", 1, "end"],
        ["body", "edit", "segments", 2, "end"],
    ]
    assert r.json()["detail"][0]["msg"] == "must be within the video (0-120.00 s)"


def test_limits_that_are_allowed(
    create: CreateFn, video: tuple[uuid.UUID, list[int]]
) -> None:
    """Edges: 200 segments, 0.3 s ones, overlapping ones (replays) in any order, the
    video's end (with the 0.1 s player tolerance), a 64-character id, footage
    outside the rallies (rally_id null); 20 overlays, 80 characters of text, the
    smallest and largest size, positions on the frame's edges, overlapping ones,
    0.3 s ones."""
    a, _ = video[1]
    edit = {
        "segments": [
            seg("x" * SEGMENT_ID_MAX_LENGTH, 30, 30.3),
            seg("end", 119.5, DURATION + 0.05),
            seg("rally", 9, 21, rally_id=a),
            seg("replay", 12, 14, slow_mo=True, rally_id=a),
            *many(MAX_REEL_SEGMENTS - 4, length=0.3),
        ],
        "overlays": [
            overlay(
                "x" * OVERLAY_ID_MAX_LENGTH, 0, 10, text="x" * OVERLAY_TEXT_MAX_LENGTH
            ),
            overlay("small", 0, 0.3, size=OVERLAY_SIZE_MIN, x=0, y=0),
            overlay("big", 0, 10, size=OVERLAY_SIZE_MAX, x=1, y=1),
            *[overlay(f"o{i}", 1, 2) for i in range(MAX_OVERLAYS - 3)],
        ],
    }
    reel = create(edit=edit)
    assert len(reel["edit"]["segments"]) == MAX_REEL_SEGMENTS
    assert reel["edit"]["overlays"] == edit["overlays"]
    assert reel["edit"]["segments"][1]["end"] == DURATION  # stored to 0.1 s


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
            "edit": {"segments": [seg("a", 100, 130)]},
            "updated_at": reel["updated_at"],
        },
    )
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "edit", "segments", 0, "end"]
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
    assert row.edit["version"] == 6
    assert row.edit["segments"][2]["slow_mo"] is True
