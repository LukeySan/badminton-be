"""Export rows: creating (one active per reel, per-user limit, race-safe), the worker
job functions (claim, heartbeat, finish, fail, release, stale re-queue), the soft
delete and the purge."""

import threading
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, col, delete, func, update

from app import crud
from app.core.config import settings
from app.export_snapshot import build_snapshot
from app.models import (
    Export,
    ExportFailureReason,
    Reel,
    ReelEdit,
    Video,
    export_filename,
)
from app.purge import purge
from tests.crud.test_video_jobs import (  # noqa: F401  (empty_videos is a fixture)
    empty_videos,
    make_video,
)

EDIT = ReelEdit.model_validate(
    {"clips": [{"rally_id": 1, "pieces": [{"start": 1.0, "end": 3.0}]}]}
)


def make_reel(db: Session, video_id: uuid.UUID, title: str = "Highlights") -> Reel:
    video = db.get(Video, video_id)
    assert video
    reel = crud.create_reel(session=db, video=video, title=title, edit=EDIT)
    assert reel
    return reel


def new_export(
    db: Session, video_id: uuid.UUID | None = None, max_active: int = 3
) -> tuple[Export | None, str]:
    video_id = video_id or make_video(db, status="done")
    video = db.get(Video, video_id)
    assert video
    reel = crud.get_video_reel(session=db, video_id=video_id) or make_reel(db, video_id)
    return crud.create_export(
        session=db,
        video=video,
        reel=reel,
        snapshot=build_snapshot(EDIT, duration=60.0),
        quality="720p",
        max_active=max_active,
    )


def make_export(db: Session, video_id: uuid.UUID | None = None) -> Export:
    export, result = new_export(db, video_id)
    assert result == "created" and export
    return export


def fetch_export(engine: Engine, export_id: uuid.UUID) -> Export:
    with Session(engine) as session:
        export = session.get(Export, export_id)
        assert export
        session.expunge(export)
        return export


def set_export_age(db: Session, export_id: uuid.UUID, age: timedelta) -> None:
    """Make updated_at (queue time / heartbeat) `age` ago, on the DB clock."""
    db.exec(
        update(Export)
        .where(col(Export.id) == export_id)
        .values(updated_at=func.now() - age)
    )
    db.commit()


# --- creating -------------------------------------------------------------------


def test_create_stores_the_snapshot_and_the_reel_title(db: Session) -> None:
    video_id = make_video(db, status="done")
    reel = make_reel(db, video_id, title="Saturday doubles")
    export = make_export(db, video_id)
    assert (export.status, export.reel_id, export.title) == (
        "queued",
        reel.id,
        "Saturday doubles",
    )
    assert export.snapshot == build_snapshot(EDIT, duration=60.0).model_dump(
        mode="json"
    )
    assert export.owner_id == reel.owner_id


def test_one_active_export_per_reel(db: Session) -> None:
    video_id = make_video(db, status="done")
    first = make_export(db, video_id)
    assert new_export(db, video_id) == (None, "already_active")
    # Finished (or failed): the reel can be exported again
    db.exec(update(Export).where(col(Export.id) == first.id).values(status="done"))
    db.commit()
    assert new_export(db, video_id)[1] == "created"


def test_deleted_active_export_doesnt_block(db: Session) -> None:
    video_id = make_video(db, status="done")
    first = make_export(db, video_id)
    assert crud.soft_delete_export(session=db, export_id=first.id) == ""
    assert new_export(db, video_id)[1] == "created"


def test_per_user_limit(db: Session) -> None:
    for _ in range(2):
        make_export(db)
    assert new_export(db, max_active=2) == (None, "limit_reached")
    # "Already exporting" wins when both apply
    video_id = make_video(db, status="done")
    make_reel(db, video_id)
    assert new_export(db, video_id, max_active=3)[1] == "created"
    assert new_export(db, video_id, max_active=3) == (None, "already_active")


def test_simultaneous_creates_queue_one(db: Session, engine: Engine) -> None:
    """Six "Export" clicks at once: the user-row lock and the partial unique index
    let exactly one through."""
    video_id = make_video(db, status="done")
    make_reel(db, video_id)
    results: list[str] = []
    barrier = threading.Barrier(6)

    def click() -> None:
        with Session(engine) as session:
            video = session.get(Video, video_id)
            reel = crud.get_video_reel(session=session, video_id=video_id)
            assert video and reel
            barrier.wait()
            results.append(
                crud.create_export(
                    session=session,
                    video=video,
                    reel=reel,
                    snapshot=build_snapshot(EDIT, duration=60.0),
                    quality="720p",
                    max_active=3,
                )[1]
            )

    threads = [threading.Thread(target=click) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["already_active"] * 5 + ["created"]


@pytest.mark.parametrize(
    ("title", "filename"),
    [
        ("Saturday doubles", "saturday-doubles-highlight.mp4"),
        ("  Rally #5: what a smash!!  ", "rally-5-what-a-smash-highlight.mp4"),
        ("แมตช์วันเสาร์", "แมตช์วันเสาร์-highlight.mp4"),  # marks kept with letters
        ("!!!", "highlight.mp4"),
        ("clipped3_vid highlights", "clipped3-vid-highlights.mp4"),
        ("Best Highlight", "best-highlight.mp4"),
        ("Highlightsfest", "highlightsfest-highlight.mp4"),
        ("x" * 100, "x" * 80 + "-highlight.mp4"),
    ],
)
def test_export_filename(title: str, filename: str) -> None:
    assert export_filename(title) == filename


# --- worker jobs ------------------------------------------------------------------


def test_claim_oldest_and_skip_deleted(db: Session, engine: Engine) -> None:
    deleted = make_export(db)
    crud.soft_delete_export(session=db, export_id=deleted.id)
    older = make_export(db)
    set_export_age(db, older.id, timedelta(minutes=5))
    newer = make_export(db)

    with Session(engine) as session:
        claimed = crud.claim_next_export(session=session, claimed_by="w")
        assert claimed and claimed.id == older.id
        assert (claimed.status, claimed.attempts) == ("processing", 1)
    with Session(engine) as session:
        claimed = crud.claim_next_export(session=session, claimed_by="w")
        assert claimed and claimed.id == newer.id
    with Session(engine) as session:
        assert crud.claim_next_export(session=session, claimed_by="w") is None


def claim(engine: Engine) -> Export:
    with Session(engine) as session:
        export = crud.claim_next_export(session=session, claimed_by="w")
        assert export and export.claim_id
        session.expunge(export)
        return export


def test_heartbeat_finish_and_lost_after_delete(db: Session, engine: Engine) -> None:
    make_export(db)
    export = claim(engine)
    assert export.claim_id
    ids = {"export_id": export.id, "claim_id": export.claim_id}
    assert crud.heartbeat_export(session=db, progress=0.5, **ids)
    assert fetch_export(engine, export.id).progress == 0.5

    assert crud.soft_delete_export(session=db, export_id=export.id) == ""
    assert not crud.heartbeat_export(session=db, progress=0.6, **ids)
    assert not crud.finish_export(
        session=db, output_path="x.mp4", size_bytes=1, duration=1.0, **ids
    )
    assert fetch_export(engine, export.id).output_path is None


def test_finish_fail_and_release(db: Session, engine: Engine) -> None:
    for _ in range(3):
        make_export(db)
    done, failed, released = claim(engine), claim(engine), claim(engine)

    assert crud.finish_export(session=db, export_id=done.id, claim_id=done.claim_id,  # type: ignore[arg-type]
                              output_path="v/exports/e.mp4", size_bytes=1234,
                              duration=12.5)  # fmt: skip
    row = fetch_export(engine, done.id)
    assert (row.status, row.progress, row.output_path, row.size_bytes, row.duration,
            row.claim_id) == ("done", 1.0, "v/exports/e.mp4", 1234, 12.5, None)  # fmt: skip

    assert crud.fail_export(session=db, export_id=failed.id, claim_id=failed.claim_id,  # type: ignore[arg-type]
                            reason=ExportFailureReason.UNKNOWN, error="RenderFailed: x")  # fmt: skip
    row = fetch_export(engine, failed.id)
    assert (row.status, row.failure_reason, row.error) == (
        "failed",
        ExportFailureReason.UNKNOWN,
        "RenderFailed: x",
    )

    assert crud.release_export(session=db, export_id=released.id, claim_id=released.claim_id)  # type: ignore[arg-type]  # fmt: skip
    row = fetch_export(engine, released.id)
    assert (row.status, row.attempts, row.claim_id) == ("queued", 0, None)


def test_stale_exports_are_requeued_then_given_up(db: Session, engine: Engine) -> None:
    retry, give_up = make_export(db), make_export(db)
    for export, attempts in ((retry, 1), (give_up, 3)):
        db.exec(
            update(Export)
            .where(col(Export.id) == export.id)
            .values(status="processing", attempts=attempts, claim_id=uuid.uuid4())
        )
        db.commit()
        set_export_age(db, export.id, timedelta(minutes=10))

    assert crud.requeue_stale_exports(session=db, stale_after=timedelta(minutes=2)) == (
        1,
        1,
    )
    assert fetch_export(engine, retry.id).status == "queued"
    gone = fetch_export(engine, give_up.id)
    assert (gone.status, gone.failure_reason) == (
        "failed",
        ExportFailureReason.KEPT_STOPPING,
    )


# --- purge ------------------------------------------------------------------------


def with_file(db: Session, export: Export, age: timedelta) -> Path:
    relative = f"{export.video_id}/exports/{export.id}.mp4"
    path = settings.STORAGE_DIR / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"mp4")
    db.exec(
        update(Export)
        .where(col(Export.id) == export.id)
        .values(status="done", output_path=relative, created_at=func.now() - age)
    )
    db.commit()
    return path


def test_purge_deletes_expired_exports_and_their_files(
    db: Session, engine: Engine
) -> None:
    db.exec(delete(Export))
    db.commit()
    expired, fresh, rendering = make_export(db), make_export(db), make_export(db)
    old_file = with_file(db, expired, timedelta(days=8))
    fresh_file = with_file(db, fresh, timedelta(days=6))
    db.exec(
        update(Export)
        .where(col(Export.id) == rendering.id)
        .values(status="processing", created_at=func.now() - timedelta(days=8))
    )
    db.commit()
    # Deleted by the user, file still there (e.g. the API couldn't remove it)
    user_deleted = make_export(db)
    leftover = with_file(db, user_deleted, timedelta(hours=1))
    db.exec(
        update(Export)
        .where(col(Export.id) == user_deleted.id)
        .values(deleted_at=func.now())
    )
    db.commit()

    assert purge(db) == (1, 2)
    assert not old_file.exists() and not leftover.exists()
    assert fresh_file.exists()
    gone = fetch_export(engine, expired.id)
    assert gone.deleted_at is not None and gone.output_path is None
    assert fetch_export(engine, fresh.id).deleted_at is None
    assert fetch_export(engine, rendering.id).deleted_at is None  # still running
    assert purge(db) == (0, 0)  # nothing left to do
