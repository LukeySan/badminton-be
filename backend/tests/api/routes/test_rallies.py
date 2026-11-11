"""Rally corrections: add, edit, delete, split, merge."""

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, col, select, update

from app import crud
from app.core.config import settings
from app.models import Correction, Rally, UserCreate, Video
from tests.utils.user import user_authentication_headers
from tests.utils.utils import random_email, random_lower_string

API = settings.API_V1_STR
DURATION = 120.0
SPANS = [(10.0, 20.0), (30.0, 45.0), (60.0, 70.0)]  # the detector's rallies


@pytest.fixture
def video(db: Session) -> tuple[uuid.UUID, list[int]]:
    """The normal user's analyzed video: (id, rally ids in time order)."""
    owner = crud.get_user_by_email(session=db, email=settings.EMAIL_TEST_USER)
    assert owner
    video = Video(
        owner_id=owner.id,
        filename="match.mp4",
        storage_path="x/original.mp4",
        status="done",
        duration=DURATION,
    )
    db.add(video)
    db.commit()
    rallies = [
        Rally(video_id=video.id, start=s, end=e, detected_start=s, detected_end=e)
        for s, e in SPANS
    ]
    db.add_all(rallies)
    db.commit()
    return video.id, [r.id for r in rallies if r.id is not None]


def url(video_id: uuid.UUID, path: str = "") -> str:
    return f"{API}/videos/{video_id}/rallies{path}"


def spans(
    change: dict[str, Any], key: str = "rallies"
) -> list[tuple[int, float, float]]:
    return [(r["number"], r["start"], r["end"]) for r in change[key]]


def corrections(db: Session, video_id: uuid.UUID) -> list[Correction]:
    db.expire_all()
    return list(
        db.exec(
            select(Correction)
            .where(col(Correction.video_id) == video_id)
            .order_by(col(Correction.id))
        ).all()
    )


def make_reel(
    client: TestClient,
    headers: dict[str, str],
    video_id: uuid.UUID,
    clips: list[tuple[int, list[tuple[float, float]]]],
) -> dict[str, Any]:
    edit = {
        "clips": [
            {"rally_id": r, "pieces": [{"start": s, "end": e} for s, e in pieces]}
            for r, pieces in clips
        ]
    }
    r = client.post(
        f"{API}/videos/{video_id}/reels",
        headers=headers,
        json={"title": "Highlights", "edit": edit},
    )
    assert r.status_code == 201, r.text
    reel: dict[str, Any] = r.json()
    return reel


def reel_clips(
    client: TestClient, headers: dict[str, str], reel_id: str
) -> list[tuple[int, list[tuple[float, float]]]]:
    r = client.get(f"{API}/reels/{reel_id}", headers=headers)
    return [
        (c["rally_id"], [(p["start"], p["end"]) for p in c["pieces"]])
        for c in r.json()["edit"]["clips"]
    ]


# --- the list -----------------------------------------------------------------


def test_rallies_are_numbered_in_time_order(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    r = client.get(url(video[0]), headers=normal_user_token_headers)
    assert [(d["number"], d["id"]) for d in r.json()["data"]] == [
        (1, video[1][0]),
        (2, video[1][1]),
        (3, video[1][2]),
    ]


# --- add ----------------------------------------------------------------------


def test_add_rally(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    r = client.post(
        url(video[0]), headers=normal_user_token_headers, json={"start": 20, "end": 28}
    )
    assert r.status_code == 201, r.text
    change = r.json()
    # Inserted between rallies 1 and 2: the later ones are renumbered
    assert spans(change) == [(1, 10, 20), (2, 20, 28), (3, 30, 45), (4, 60, 70)]
    assert spans(change, "added") == [(2, 20, 28)]
    assert change["updated"] == change["deleted"] == change["reels"] == []

    new_id = change["added"][0]["id"]
    added = db.get(Rally, new_id)
    assert added and added.source == "user" and added.detected_start is None
    (log,) = corrections(db, video[0])
    assert (log.action, log.rally_id, log.result_rally_id) == ("add", None, new_id)
    assert (log.original_start, log.original_end) == (None, None)  # not detected
    assert (log.edited_start, log.edited_end) == (20, 28)
    owner = crud.get_user_by_email(session=db, email=settings.EMAIL_TEST_USER)
    assert owner and log.user_id == owner.id


@pytest.mark.parametrize(
    ("times", "status"),
    [
        ({"start": 15, "end": 25}, 409),  # overlaps rally 1
        ({"start": 25, "end": 75}, 409),  # contains rallies 2 and 3
        ({"start": 100, "end": 121}, 422),  # past the end of the video
        ({"start": 25, "end": 25.05}, 422),  # shorter than 0.1 s
        ({"start": 26, "end": 25}, 422),
        ({"start": -1, "end": 5}, 422),
    ],
)
def test_add_refused(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    times: dict[str, float],
    status: int,
) -> None:
    r = client.post(url(video[0]), headers=normal_user_token_headers, json=times)
    assert r.status_code == status, r.text
    if status == 409:
        assert r.json()["detail"].startswith("Overlaps rally ")
    assert corrections(db, video[0]) == []


# --- edit ---------------------------------------------------------------------


def test_edit_rally_trims_reel_pieces(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h, (a, b, _) = normal_user_token_headers, video[1]
    reel = make_reel(client, h, video[0], [(a, [(9.5, 12), (16, 21)]), (b, [(31, 33)])])
    r = client.put(url(video[0], f"/{a}"), headers=h, json={"start": 11, "end": 18})
    assert r.status_code == 200, r.text
    change = r.json()
    assert spans(change, "updated") == [(1, 11, 18)]
    # Pieces may reach 10-19 now (1 s lead): both pieces trimmed
    assert reel_clips(client, h, reel["id"]) == [(a, [(10, 12), (16, 19)]), (b, [(31, 33)])]  # fmt: skip
    (impact,) = change["reels"]
    assert impact["id"] == reel["id"]
    assert (impact["clips_removed"], impact["pieces_trimmed"], impact["pieces_removed"]) == (0, 2, 0)  # fmt: skip
    assert impact["updated_at"] > reel["updated_at"]

    # An editor still holding the old version must reload
    r = client.put(
        f"{API}/reels/{reel['id']}",
        headers=h,
        json={"title": "x", "edit": {}, "updated_at": reel["updated_at"]},
    )
    assert r.status_code == 409 and r.json()["code"] == "reel_changed"


def test_corrections_keep_the_detectors_original(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    """Edited twice: both log rows hold the detector's 10-20, not the previous
    edit. That's what makes them usable as training labels."""
    h, a = normal_user_token_headers, video[1][0]
    client.put(url(video[0], f"/{a}"), headers=h, json={"start": 11, "end": 19})
    client.put(url(video[0], f"/{a}"), headers=h, json={"start": 12, "end": 18})
    logs = corrections(db, video[0])
    assert [(c.action, c.original_start, c.original_end, c.edited_start, c.edited_end) for c in logs] == [
        ("edit", 10, 20, 11, 19),
        ("edit", 10, 20, 12, 18),
    ]  # fmt: skip
    assert logs[0].operation_id != logs[1].operation_id


def test_edit_to_the_same_times_changes_nothing(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    a = video[1][0]
    r = client.put(
        url(video[0], f"/{a}"),
        headers=normal_user_token_headers,
        json={"start": 10, "end": 20},
    )
    assert r.status_code == 200
    assert r.json()["updated"] == []
    assert corrections(db, video[0]) == []


def test_edit_may_grow_into_the_break_but_not_into_another_rally(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h, b = normal_user_token_headers, video[1][1]
    r = client.put(url(video[0], f"/{b}"), headers=h, json={"start": 20, "end": 60})
    assert r.status_code == 200  # touching rallies 1 and 3 is fine
    r = client.put(url(video[0], f"/{b}"), headers=h, json={"start": 19, "end": 45})
    assert r.status_code == 409
    assert r.json()["detail"] == "Overlaps rally 1 (10.00-20.00 s)"


# --- delete -------------------------------------------------------------------


def test_delete_rally_drops_its_reel_clips(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h, (a, b, c) = normal_user_token_headers, video[1]
    reel = make_reel(client, h, video[0], [(a, [(10, 12)]), (b, [(31, 33)])])
    untouched = make_reel(client, h, video[0], [(c, [(61, 62)])])
    r = client.delete(url(video[0], f"/{a}"), headers=h)
    assert r.status_code == 200, r.text
    change = r.json()
    assert change["deleted"] == [a]
    assert spans(change) == [(1, 30, 45), (2, 60, 70)]
    assert [(i["id"], i["clips_removed"]) for i in change["reels"]] == [(reel["id"], 1)]
    assert reel_clips(client, h, reel["id"]) == [(b, [(31, 33)])]
    r = client.get(f"{API}/reels/{untouched['id']}", headers=h)
    assert r.json()["updated_at"] == untouched["updated_at"]

    (log,) = corrections(db, video[0])
    assert (log.action, log.rally_id, log.result_rally_id) == ("delete", a, None)
    assert (log.original_start, log.original_end) == (10, 20)
    assert (log.edited_start, log.edited_end) == (None, None)
    # The log outlives the rally
    assert client.delete(url(video[0], f"/{a}"), headers=h).status_code == 404


# --- split --------------------------------------------------------------------


def test_split_rally(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h, (a, b, _) = normal_user_token_headers, video[1]
    reel = make_reel(client, h, video[0], [(a, [(10, 12), (13, 17)]), (b, [(31, 33)])])
    r = client.post(url(video[0], f"/{a}/split"), headers=h, json={"at": 15})
    assert r.status_code == 200, r.text
    change = r.json()
    second = change["added"][0]["id"]
    assert spans(change, "updated") == [(1, 10, 15)]
    assert change["updated"][0]["id"] == a  # the first part keeps the id
    assert spans(change, "added") == [(2, 15, 20)]
    assert spans(change) == [(1, 10, 15), (2, 15, 20), (3, 30, 45), (4, 60, 70)]
    # The clip is split the same way, nothing lost
    assert reel_clips(client, h, reel["id"]) == [
        (a, [(10, 12), (13, 15)]),
        (second, [(15, 17)]),
        (b, [(31, 33)]),
    ]
    impact = change["reels"][0]
    assert (impact["clips_removed"], impact["pieces_trimmed"], impact["pieces_removed"]) == (0, 0, 0)  # fmt: skip

    logs = corrections(db, video[0])
    assert [(c.action, c.rally_id, c.result_rally_id, c.edited_start, c.edited_end) for c in logs] == [
        ("split", a, a, 10, 15),
        ("split", a, second, 15, 20),
    ]  # fmt: skip
    assert {(c.original_start, c.original_end) for c in logs} == {(10, 20)}
    assert logs[0].operation_id == logs[1].operation_id
    # Both parts remember the detector's rally they came from
    part = db.get(Rally, second)
    assert part and (part.detected_start, part.detected_end) == (10, 20)


@pytest.mark.parametrize("at", [10, 10.05, 19.95, 20, 25])
def test_split_point_must_be_inside_the_rally(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    at: float,
) -> None:
    r = client.post(
        url(video[0], f"/{video[1][0]}/split"),
        headers=normal_user_token_headers,
        json={"at": at},
    )
    assert r.status_code == 422
    assert "inside rally 1 (10.00-20.00 s)" in r.json()["detail"]


# --- merge --------------------------------------------------------------------


def test_merge_rallies(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h, (a, b, c) = normal_user_token_headers, video[1]
    reel = make_reel(
        client, h, video[0], [(a, [(15, 21)]), (b, [(29, 33)]), (c, [(61, 62)])]
    )
    # Either order
    r = client.post(url(video[0], "/merge"), headers=h, json={"rally_ids": [b, a]})
    assert r.status_code == 200, r.text
    change = r.json()
    merged = change["added"][0]["id"]
    assert merged not in (a, b)
    assert change["deleted"] == sorted([a, b])
    assert spans(change) == [(1, 10, 45), (2, 60, 70)]
    # Back-to-back clips of the two became one clip of the merged rally
    assert reel_clips(client, h, reel["id"]) == [
        (merged, [(15, 21), (29, 33)]),
        (c, [(61, 62)]),
    ]

    logs = corrections(db, video[0])
    assert [(x.action, x.rally_id, x.result_rally_id, x.original_start, x.original_end) for x in logs] == [
        ("merge", a, merged, 10, 20),
        ("merge", b, merged, 30, 45),
    ]  # fmt: skip
    assert {(x.edited_start, x.edited_end) for x in logs} == {(10, 45)}
    rally = db.get(Rally, merged)
    assert rally and (rally.detected_start, rally.detected_end) == (10, 45)


def test_merge_with_an_added_rally_keeps_the_detected_side(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h, a = normal_user_token_headers, video[1][0]
    added = client.post(url(video[0]), headers=h, json={"start": 21, "end": 25})
    new_id = added.json()["added"][0]["id"]
    r = client.post(url(video[0], "/merge"), headers=h, json={"rally_ids": [a, new_id]})
    merged = db.get(Rally, r.json()["added"][0]["id"])
    assert merged and (merged.detected_start, merged.detected_end) == (10, 20)


@pytest.mark.parametrize(
    ("pick", "status", "detail"),
    [
        ((0, 2), 409, "Rallies 1 and 3 aren't next to each other"),
        ((0, 0), 422, "Give two different rallies"),
    ],
)
def test_merge_refused(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    pick: tuple[int, int],
    status: int,
    detail: str,
) -> None:
    ids = [video[1][i] for i in pick]
    r = client.post(
        url(video[0], "/merge"),
        headers=normal_user_token_headers,
        json={"rally_ids": ids},
    )
    assert r.status_code == status
    assert r.json()["detail"].startswith(detail)
    assert corrections(db, video[0]) == []


def test_merge_needs_exactly_two(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    r = client.post(
        url(video[0], "/merge"),
        headers=normal_user_token_headers,
        json={"rally_ids": video[1]},
    )
    assert r.status_code == 422


# --- refusals and access ------------------------------------------------------


@pytest.mark.parametrize("status", ["queued", "processing"])
def test_no_corrections_while_queued_or_processing(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    status: str,
) -> None:
    """Re-analysis replaces all rallies, which would silently undo the correction."""
    db.exec(update(Video).where(col(Video.id) == video[0]).values(status=status))
    db.commit()
    h, a = normal_user_token_headers, video[1][0]
    for r in [
        client.post(url(video[0]), headers=h, json={"start": 21, "end": 25}),
        client.put(url(video[0], f"/{a}"), headers=h, json={"start": 11, "end": 19}),
        client.delete(url(video[0], f"/{a}"), headers=h),
        client.post(url(video[0], f"/{a}/split"), headers=h, json={"at": 15}),
    ]:
        assert r.status_code == 409
        assert r.json()["detail"] == f"Can't change rallies while the video is {status}"
    assert corrections(db, video[0]) == []


def test_rally_of_another_video_is_not_found(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    other = Video(
        owner_id=db.get(Video, video[0]).owner_id,  # type: ignore[union-attr]
        filename="b.mp4",
        storage_path="y/original.mp4",
        status="done",
        duration=60.0,
    )
    db.add(other)
    db.commit()
    h = normal_user_token_headers
    a = video[1][0]
    assert client.delete(url(other.id, f"/{a}"), headers=h).status_code == 404
    r = client.post(
        url(other.id, "/merge"), headers=h, json={"rally_ids": video[1][:2]}
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "Rally not found"


@pytest.mark.usefixtures("normal_user_token_headers")  # creates the video owner
def test_other_users_get_404(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
) -> None:
    email, password = random_email(), random_lower_string()
    crud.create_user(session=db, user_create=UserCreate(email=email, password=password))
    h = user_authentication_headers(client=client, email=email, password=password)
    a = video[1][0]
    assert client.delete(url(video[0], f"/{a}"), headers=h).status_code == 404
    r = client.post(url(video[0]), headers=h, json={"start": 21, "end": 25})
    assert r.status_code == 404
    assert corrections(db, video[0]) == []


def test_superuser_corrections_are_logged_as_theirs(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    superuser_token_headers: dict[str, str],
) -> None:
    r = client.delete(url(video[0], f"/{video[1][0]}"), headers=superuser_token_headers)
    assert r.status_code == 200
    admin = crud.get_user_by_email(session=db, email=settings.FIRST_SUPERUSER)
    assert admin and corrections(db, video[0])[0].user_id == admin.id


def test_missing_video(
    client: TestClient, normal_user_token_headers: dict[str, str]
) -> None:
    r = client.post(
        url(uuid.uuid4()),
        headers=normal_user_token_headers,
        json={"start": 1, "end": 2},
    )
    assert r.status_code == 404
