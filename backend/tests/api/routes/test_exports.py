"""Reel exports API: POST/GET /videos/{id}/reel/exports, GET/DELETE /exports/{id},
the signed download link and the file."""

import uuid
from collections.abc import Generator
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, col, delete, func, update

from app import crud
from app.core.config import settings
from app.core.security import create_access_token, create_file_token
from app.models import Export, UserCreate, Video
from tests.api.routes.test_reels import make_video
from tests.utils.user import user_authentication_headers
from tests.utils.utils import random_email, random_lower_string

API = settings.API_V1_STR


@pytest.fixture(autouse=True)
def no_exports(db: Session) -> Generator[None]:
    """The per-user limit counts every queued export: start each test without."""
    db.exec(delete(Export))
    db.commit()
    yield


@pytest.fixture(scope="module")
def other_user_headers(client: TestClient, db: Session) -> dict[str, str]:
    email, password = random_email(), random_lower_string()
    crud.create_user(session=db, user_create=UserCreate(email=email, password=password))
    return user_authentication_headers(client=client, email=email, password=password)


@pytest.fixture
def video(db: Session) -> tuple[uuid.UUID, list[int]]:
    """The normal user's video (120 s, rallies 10-20 and 30-45 s), 1280x720."""
    video_id, rallies = make_video(db, settings.EMAIL_TEST_USER)
    set_size(db, video_id, 1280, 720)
    return video_id, rallies


def set_size(db: Session, video_id: uuid.UUID, width: int, height: int) -> None:
    db.exec(
        update(Video)
        .where(col(Video.id) == video_id)
        .values(width=width, height=height)
    )
    db.commit()


def seg(
    id: str,
    start: float,
    end: float,
    slow_mo: bool = False,
    rally_id: int | None = None,
) -> dict[str, Any]:
    return {"id": id, "start": start, "end": end, "slow_mo": slow_mo,
            "rally_id": rally_id}  # fmt: skip


def save_reel(
    client: TestClient,
    headers: dict[str, str],
    video_id: uuid.UUID,
    edit: dict[str, Any],
    title: str = "Saturday doubles",
) -> dict[str, Any]:
    current = client.get(f"{API}/videos/{video_id}/reel", headers=headers)
    updated_at = current.json()["updated_at"] if current.status_code == 200 else None
    r = client.put(
        f"{API}/videos/{video_id}/reel",
        headers=headers,
        json={"title": title, "edit": edit, "updated_at": updated_at},
    )
    assert r.status_code in (200, 201), r.text
    return dict(r.json())


def reel_edit(rallies: list[int]) -> dict[str, Any]:
    a, b = rallies
    return {
        "format": "9:16",
        "segments": [
            seg("a1", 9, 15, rally_id=a),
            seg("a2", 15, 18, slow_mo=True, rally_id=a),
            seg("break", 18, 29),  # footage between the rallies
            seg("b", 29, 46, rally_id=b),
        ],
        # Reel time: over the end of a1 and the slow-mo a2 (reel 6-12)
        "overlays": [
            {
                "id": "o1",
                "type": "text",
                "text": "สุดยอด!",
                "start": 3,
                "end": 8,
                "x": 0.3,
                "y": 0.2,
                "size": 0.1,
            }
        ],
    }


def export_url(video_id: uuid.UUID) -> str:
    return f"{API}/videos/{video_id}/reel/exports"


def start_export(
    client: TestClient,
    headers: dict[str, str],
    video_id: uuid.UUID,
    **body: Any,
) -> Any:
    return client.post(export_url(video_id), headers=headers, json=body)


def finish(db: Session, export_id: str, content: bytes = b"fake mp4 bytes") -> None:
    """Mark the export done with a file, as the worker would."""
    export = db.get(Export, uuid.UUID(export_id))
    assert export
    relative = f"{export.video_id}/exports/{export.id}.mp4"
    path = settings.STORAGE_DIR / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    db.exec(
        update(Export)
        .where(col(Export.id) == export.id)
        .values(
            status="done",
            progress=1.0,
            output_path=relative,
            size_bytes=len(content),
            duration=31.07,
        )
    )
    db.commit()


# --- creating -------------------------------------------------------------------


def test_create_export(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    reel = save_reel(client, h, video[0], reel_edit(video[1]))
    r = start_export(client, h, video[0])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["video_id"] == str(video[0]) and body["reel_id"] == reel["id"]
    assert (body["status"], body["progress"], body["failure_reason"]) == (
        "queued",
        0.0,
        None,
    )
    # The reel's format; 720p by default
    assert (body["format"], body["quality"], body["width"], body["height"]) == (
        "9:16",
        "720p",
        720,
        1280,
    )
    # 6 + 3 * 2 (slow-mo) + 11 (the break 18-29) + 17 seconds
    assert body["duration"] == 40.0
    assert body["size_bytes"] is None
    assert body["filename"] == "saturday-doubles-highlight.mp4"
    created = datetime.fromisoformat(body["created_at"])
    assert datetime.fromisoformat(body["expires_at"]) - created == timedelta(days=7)

    stored = db.get(Export, uuid.UUID(body["id"]))
    assert stored
    db.refresh(stored)
    a, b = video[1]
    assert [s["rally_id"] for s in stored.snapshot["segments"]] == [a, a, None, b]
    assert stored.snapshot["version"] == 2
    assert stored.snapshot["overlays"] == [{"type": "text", "text": "สุดยอด!",
        "start": 3.0, "end": 8.0, "x": 0.3, "y": 0.2, "size": 0.1}]  # fmt: skip


def test_format_override_and_1080p(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    save_reel(client, h, video[0], reel_edit(video[1]))
    r = start_export(client, h, video[0], quality="1080p")
    assert r.status_code == 422
    assert "1080p needs a video of at least 1080p; this one is 1280x720" in r.text

    set_size(db, video[0], 1920, 1080)
    r = start_export(client, h, video[0], quality="1080p", format="16:9")
    assert r.status_code == 201, r.text
    assert (r.json()["format"], r.json()["width"], r.json()["height"]) == (
        "16:9",
        1920,
        1080,
    )


def test_portrait_source_counts_its_short_side(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    set_size(db, video[0], 1080, 1920)
    save_reel(client, normal_user_token_headers, video[0], reel_edit(video[1]))
    r = start_export(client, normal_user_token_headers, video[0], quality="1080p")
    assert r.status_code == 201, r.text
    assert (r.json()["width"], r.json()["height"]) == (1080, 1920)


def test_no_reel_or_no_pieces(
    client: TestClient,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    r = start_export(client, h, video[0])
    assert r.status_code == 404
    assert r.json()["code"] == "reel_missing"
    # An empty reel isn't something to export
    save_reel(client, h, video[0], {})
    r = start_export(client, h, video[0])
    assert r.status_code == 409
    assert r.json()["detail"] == "The reel has no clips yet; add some to export it"


def test_one_active_export_per_reel(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    save_reel(client, h, video[0], reel_edit(video[1]))
    first = start_export(client, h, video[0])
    assert first.status_code == 201
    again = start_export(client, h, video[0], quality="720p", format="16:9")
    assert again.status_code == 409
    assert "already being exported" in again.json()["detail"]
    finish(db, first.json()["id"])
    assert start_export(client, h, video[0]).status_code == 201


def test_per_user_limit(
    client: TestClient,
    db: Session,
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    for _ in range(settings.MAX_ACTIVE_EXPORTS_PER_USER):
        video_id, rallies = make_video(db, settings.EMAIL_TEST_USER)
        save_reel(client, h, video_id, reel_edit(rallies))
        assert start_export(client, h, video_id).status_code == 201
    video_id, rallies = make_video(db, settings.EMAIL_TEST_USER)
    save_reel(client, h, video_id, reel_edit(rallies))
    r = start_export(client, h, video_id)
    assert r.status_code == 429
    assert r.json()["detail"] == (
        "You already have 3 exports queued or processing; try again when one has "
        "finished"
    )


def test_snapshot_unaffected_by_later_edits_and_rally_fixes(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    """What's rendered is fixed at request time: the queued export keeps it while
    the user re-edits the reel and fixes a rally."""
    h = normal_user_token_headers
    save_reel(client, h, video[0], reel_edit(video[1]))
    body = start_export(client, h, video[0]).json()
    stored = db.get(Export, uuid.UUID(body["id"]))
    assert stored
    snapshot_before = dict(stored.snapshot)

    a, b = video[1]
    save_reel(client, h, video[0], {"segments": [seg("b", 30, 31, rally_id=b)]},
              title="Renamed")  # fmt: skip
    r = client.put(f"{API}/videos/{video[0]}/rallies/{b}", headers=h,
                   json={"start": 32, "end": 40})  # fmt: skip
    assert r.status_code == 200, r.text

    db.refresh(stored)
    assert stored.snapshot == snapshot_before
    after = client.get(f"{API}/exports/{body['id']}", headers=h).json()
    assert (after["duration"], after["title"], after["format"]) == (40.0, "Saturday doubles", "9:16")  # fmt: skip


# --- reading, listing, permissions ------------------------------------------------


def test_list_newest_first_without_deleted(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    save_reel(client, h, video[0], reel_edit(video[1]))
    ids = []
    for _ in range(3):
        r = start_export(client, h, video[0])
        ids.append(r.json()["id"])
        finish(db, r.json()["id"])
    client.delete(f"{API}/exports/{ids[1]}", headers=h)
    listed = client.get(export_url(video[0]), headers=h).json()
    assert listed["count"] == 2
    assert [e["id"] for e in listed["data"]] == [ids[2], ids[0]]


def test_others_get_404_superusers_get_access(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    other_user_headers: dict[str, str],
    superuser_token_headers: dict[str, str],
) -> None:
    save_reel(client, normal_user_token_headers, video[0], reel_edit(video[1]))
    export_id = start_export(client, normal_user_token_headers, video[0]).json()["id"]
    finish(db, export_id)
    other = other_user_headers
    assert start_export(client, other, video[0]).status_code == 404
    assert client.get(export_url(video[0]), headers=other).status_code == 404
    for method, path in [
        ("get", f"/exports/{export_id}"),
        ("get", f"/exports/{export_id}/download"),
        ("delete", f"/exports/{export_id}"),
    ]:
        r = client.request(method, f"{API}{path}", headers=other)
        assert r.status_code == 404, path
        assert r.json() == {"detail": "Export not found"}

    su = superuser_token_headers
    assert client.get(f"{API}/exports/{export_id}", headers=su).status_code == 200
    assert client.get(export_url(video[0]), headers=su).json()["count"] == 1
    # A superuser's export belongs to the video's owner, and counts for them
    created = start_export(client, su, video[0])
    assert created.status_code == 201
    owner = crud.get_user_by_email(session=db, email=settings.EMAIL_TEST_USER)
    assert owner
    stored = db.get(Export, uuid.UUID(created.json()["id"]))
    assert stored and stored.owner_id == owner.id
    assert client.delete(f"{API}/exports/{export_id}", headers=su).status_code == 200


# --- download -------------------------------------------------------------------


def test_download_link_and_file(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    save_reel(client, h, video[0], reel_edit(video[1]))
    export_id = start_export(client, h, video[0]).json()["id"]
    r = client.get(f"{API}/exports/{export_id}/download", headers=h)
    assert r.status_code == 409  # still queued

    content = bytes(range(256)) * 10
    finish(db, export_id, content)
    link = client.get(f"{API}/exports/{export_id}/download", headers=h).json()
    assert link["filename"] == "saturday-doubles-highlight.mp4"
    assert link["url"].startswith(f"{API}/exports/{export_id}/file?token=")

    r = client.get(link["url"])  # no login header: the token is enough
    assert r.status_code == 200
    assert r.content == content
    assert r.headers["content-type"] == "video/mp4"
    assert r.headers["content-disposition"] == (
        'attachment; filename="saturday-doubles-highlight.mp4"'
    )
    r = client.get(link["url"], headers={"Range": "bytes=10-19"})
    assert (r.status_code, r.content) == (206, content[10:20])

    done = client.get(f"{API}/exports/{export_id}", headers=h).json()
    assert (done["status"], done["size_bytes"], done["duration"]) == (
        "done",
        len(content),
        31.07,
    )


def test_unicode_filename(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    save_reel(client, h, video[0], reel_edit(video[1]), title="แมตช์วันเสาร์")
    export_id = start_export(client, h, video[0]).json()["id"]
    finish(db, export_id)
    link = client.get(f"{API}/exports/{export_id}/download", headers=h).json()
    assert link["filename"] == "แมตช์วันเสาร์-highlight.mp4"
    disposition = client.get(link["url"]).headers["content-disposition"]
    assert disposition == (
        f"attachment; filename*=utf-8''{quote('แมตช์วันเสาร์-highlight.mp4')}"
    )


def test_file_tokens_are_scoped(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    save_reel(client, h, video[0], reel_edit(video[1]))
    export_id = start_export(client, h, video[0]).json()["id"]
    finish(db, export_id)
    link = client.get(f"{API}/exports/{export_id}/download", headers=h).json()
    token = link["url"].split("token=")[1]
    file_url = f"{API}/exports/{export_id}/file"

    video_token, _ = create_file_token(uuid.UUID(export_id))  # video-file scope
    other_export = create_file_token(uuid.uuid4(), scope="export-file")[0]
    login_token = create_access_token(uuid.uuid4(), timedelta(minutes=5))
    for bad in (video_token, other_export, login_token, "garbage"):
        assert client.get(f"{file_url}?token={bad}").status_code == 403
    # An export-file token is no login token either
    r = client.get(f"{API}/exports/{export_id}",
                   headers={"Authorization": f"Bearer {token}"})  # fmt: skip
    assert r.status_code == 403


def test_expired_export(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    save_reel(client, h, video[0], reel_edit(video[1]))
    export_id = start_export(client, h, video[0]).json()["id"]
    finish(db, export_id)
    link = client.get(f"{API}/exports/{export_id}/download", headers=h).json()
    # The link never outlives the export
    expires_at = client.get(f"{API}/exports/{export_id}", headers=h).json()[
        "expires_at"
    ]
    assert link["expires_at"] <= expires_at

    db.exec(
        update(Export)
        .where(col(Export.id) == uuid.UUID(export_id))
        .values(created_at=func.now() - timedelta(days=8))
    )
    db.commit()
    r = client.get(f"{API}/exports/{export_id}/download", headers=h)
    assert r.status_code == 410


def test_delete_is_soft_and_removes_the_file(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    save_reel(client, h, video[0], reel_edit(video[1]))
    export_id = start_export(client, h, video[0]).json()["id"]
    finish(db, export_id)
    link = client.get(f"{API}/exports/{export_id}/download", headers=h).json()
    stored = db.get(Export, uuid.UUID(export_id))
    assert stored and stored.output_path
    path = settings.STORAGE_DIR / stored.output_path

    r = client.delete(f"{API}/exports/{export_id}", headers=h)
    assert r.json() == {"message": "Export deleted successfully"}
    assert not path.exists()
    db.refresh(stored)
    assert stored.deleted_at is not None and stored.output_path is None
    assert client.get(f"{API}/exports/{export_id}", headers=h).status_code == 404
    assert client.get(link["url"]).status_code == 404
    assert client.delete(f"{API}/exports/{export_id}", headers=h).status_code == 404


def test_file_missing_on_disk(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    save_reel(client, h, video[0], reel_edit(video[1]))
    export_id = start_export(client, h, video[0]).json()["id"]
    finish(db, export_id)
    link = client.get(f"{API}/exports/{export_id}/download", headers=h).json()
    stored = db.get(Export, uuid.UUID(export_id))
    assert stored and stored.output_path
    (settings.STORAGE_DIR / stored.output_path).unlink()
    assert client.get(link["url"]).status_code == 404


def test_deleting_the_video_deletes_its_exports(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    save_reel(client, h, video[0], reel_edit(video[1]))
    export_id = start_export(client, h, video[0]).json()["id"]
    finish(db, export_id)
    assert client.delete(f"{API}/videos/{video[0]}", headers=h).status_code == 200
    db.expire_all()
    assert db.get(Export, uuid.UUID(export_id)) is None
    assert not (settings.STORAGE_DIR / str(video[0])).exists()


# --- trying a failed export again ---------------------------------------------------


def fail(db: Session, export_id: str) -> None:
    """Mark the export failed, as the worker would."""
    db.exec(
        update(Export)
        .where(col(Export.id) == uuid.UUID(export_id))
        .values(
            status="failed",
            progress=0.4,
            failure_reason="The export kept stopping",
            error="ffmpeg died",
            attempts=3,
        )
    )
    db.commit()


def test_retry_a_failed_export_renders_the_same_snapshot(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    save_reel(client, h, video[0], reel_edit(video[1]))
    first = start_export(client, h, video[0]).json()
    fail(db, first["id"])
    # The reel changes after the export was requested
    save_reel(client, h, video[0], {**reel_edit(video[1]), "format": "16:9"})

    r = client.post(f"{API}/exports/{first['id']}/retry", headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["status"], body["progress"], body["failure_reason"]) == (
        "queued",
        0.0,
        None,
    )
    # Same export: same edit (format, length), same request time and expiry
    for key in ("id", "format", "duration", "created_at", "expires_at"):
        assert body[key] == first[key]
    stored = db.get(Export, uuid.UUID(first["id"]))
    assert stored
    db.refresh(stored)
    assert (stored.attempts, stored.error, stored.claim_id) == (0, None, None)


def test_retry_refused(
    client: TestClient,
    db: Session,
    video: tuple[uuid.UUID, list[int]],
    normal_user_token_headers: dict[str, str],
    other_user_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    save_reel(client, h, video[0], reel_edit(video[1]))
    export_id = start_export(client, h, video[0]).json()["id"]
    retry = f"{API}/exports/{export_id}/retry"

    # Not failed (queued)
    r = client.post(retry, headers=h)
    assert r.status_code == 409
    assert r.json()["detail"] == "Only a failed export can be tried again"
    # Others don't see it
    assert client.post(retry, headers=other_user_headers).status_code == 404

    # Another export of the reel is running meanwhile
    fail(db, export_id)
    assert start_export(client, h, video[0]).status_code == 201
    r = client.post(retry, headers=h)
    assert r.status_code == 409
    assert "already being exported" in r.json()["detail"]

    # Expired
    db.exec(delete(Export).where(col(Export.id) != uuid.UUID(export_id)))
    db.exec(
        update(Export)
        .where(col(Export.id) == uuid.UUID(export_id))
        .values(created_at=func.now() - timedelta(days=8))
    )
    db.commit()
    assert client.post(retry, headers=h).status_code == 410


def test_retry_counts_against_the_per_user_limit(
    client: TestClient,
    db: Session,
    normal_user_token_headers: dict[str, str],
) -> None:
    h = normal_user_token_headers
    video_id, rallies = make_video(db, settings.EMAIL_TEST_USER)
    save_reel(client, h, video_id, reel_edit(rallies))
    failed_id = start_export(client, h, video_id).json()["id"]
    fail(db, failed_id)
    for _ in range(settings.MAX_ACTIVE_EXPORTS_PER_USER):
        other_video, other_rallies = make_video(db, settings.EMAIL_TEST_USER)
        save_reel(client, h, other_video, reel_edit(other_rallies))
        assert start_export(client, h, other_video).status_code == 201
    r = client.post(f"{API}/exports/{failed_id}/retry", headers=h)
    assert r.status_code == 429
