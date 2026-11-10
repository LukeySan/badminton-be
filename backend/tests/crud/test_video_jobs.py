import threading
import time
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import Engine, text
from sqlmodel import Session, col, delete, func, select, update

from app import crud
from app.core.config import settings
from app.models import FailureReason, Rally, Video

COURT = [[0.478, 0.347], [0.777, 0.369], [0.93, 0.943], [0.054, 0.781]]


@pytest.fixture(autouse=True)
def empty_videos(db: Session) -> None:
    """Claiming takes the oldest queued video in the whole table, so start each
    test without anyone else's videos (rallies go with them by cascade)."""
    db.exec(delete(Video))
    db.commit()


def make_video(
    db: Session,
    status: str = "queued",
    age: timedelta = timedelta(0),
    attempts: int = 0,
) -> uuid.UUID:
    owner = crud.get_user_by_email(session=db, email=settings.FIRST_SUPERUSER)
    assert owner
    video = Video(
        owner_id=owner.id,
        filename="match.mp4",
        storage_path="x/original.mp4",
        status=status,
        court_corners=COURT,
        attempts=attempts,
    )
    db.add(video)
    db.commit()
    set_age(db, video.id, age)
    return video.id


def set_age(db: Session, video_id: uuid.UUID, age: timedelta) -> None:
    """Make updated_at (queue time / last heartbeat) `age` ago, on the DB clock."""
    db.exec(
        update(Video)
        .where(col(Video.id) == video_id)
        .values(updated_at=func.now() - age)
    )
    db.commit()


def fetch(engine: Engine, video_id: uuid.UUID) -> Video:
    """Read the row in a fresh session, the way another process would see it."""
    with Session(engine) as session:
        video = session.get(Video, video_id)
        assert video
        session.expunge(video)
        return video


def rallies_of(engine: Engine, video_id: uuid.UUID) -> list[tuple[float, float]]:
    with Session(engine) as session:
        rows = session.exec(
            select(Rally).where(Rally.video_id == video_id).order_by(col(Rally.start))
        ).all()
        return [(r.start, r.end) for r in rows]


def claim(engine: Engine, name: str = "worker-a") -> Video | None:
    with Session(engine) as session:
        video = crud.claim_next_video(session=session, claimed_by=name)
        if video:
            session.expunge(video)
        return video


# --- claiming -----------------------------------------------------------------


def test_claim_takes_oldest_queued_video(db: Session, engine: Engine) -> None:
    newer = make_video(db, age=timedelta(minutes=1))
    oldest = make_video(db, age=timedelta(minutes=5))
    make_video(db, status="uploaded", age=timedelta(minutes=9))  # not queued
    make_video(db, status="done", age=timedelta(minutes=9))

    first = claim(engine, "mac:123")
    assert first and first.id == oldest
    assert first.status == "processing"
    assert first.claim_id is not None
    assert (first.claimed_by, first.attempts, first.progress) == ("mac:123", 1, 0.0)

    second = claim(engine)
    assert second and second.id == newer
    assert claim(engine) is None


def test_each_claim_gets_a_new_claim_id(db: Session, engine: Engine) -> None:
    video_id = make_video(db)
    first = claim(engine)
    assert first and first.claim_id
    with Session(engine) as session:
        crud.release_video(session=session, video_id=video_id, claim_id=first.claim_id)
    second = claim(engine)
    assert second and second.claim_id
    assert second.claim_id != first.claim_id


def test_claim_skips_a_video_another_worker_is_claiming(
    db: Session, engine: Engine
) -> None:
    """Worker A has locked the oldest video but not committed yet. Worker B must
    take the next one right away: not the same one, and without waiting."""
    oldest = make_video(db, age=timedelta(minutes=5))
    newer = make_video(db, age=timedelta(minutes=1))

    with Session(engine) as worker_a, Session(engine) as worker_b:
        # A: mid-claim, holding the row lock
        worker_a.exec(
            select(Video.id).where(col(Video.id) == oldest).with_for_update()
        ).one()
        # B: fail instead of hang if the lock ever made B wait (no SKIP LOCKED)
        worker_b.exec(text("SET lock_timeout = '2s'"))  # type: ignore[call-overload]
        started = time.monotonic()
        claimed = crud.claim_next_video(session=worker_b, claimed_by="worker-b")
        assert claimed and claimed.id == newer
        assert time.monotonic() - started < 1
        worker_a.rollback()

    assert fetch(engine, oldest).status == "queued"


def test_two_workers_never_take_the_same_job(db: Session, engine: Engine) -> None:
    """Many workers claim at the same moment, until the queue is empty: every video
    is claimed exactly once."""
    video_ids = {make_video(db, age=timedelta(seconds=i)) for i in range(8)}
    workers = 6
    start_together = threading.Barrier(workers)
    claimed: list[uuid.UUID] = []
    lock = threading.Lock()
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            start_together.wait()
            while video := claim(engine, f"worker-{n}"):
                with lock:
                    claimed.append(video.id)
        except BaseException as e:  # surface thread errors in the test
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    assert sorted(claimed) == sorted(video_ids)  # all claimed, none twice


def test_claim_on_empty_queue(engine: Engine) -> None:
    assert claim(engine) is None


# --- heartbeat, finish, fail, release ------------------------------------------


def test_heartbeat_updates_progress_and_time(db: Session, engine: Engine) -> None:
    video_id = make_video(db)
    video = claim(engine)
    assert video and video.claim_id
    set_age(db, video_id, timedelta(minutes=1))
    before = fetch(engine, video_id).updated_at

    with Session(engine) as session:
        assert crud.heartbeat_video(
            session=session, video_id=video_id, claim_id=video.claim_id, progress=0.4
        )
    after = fetch(engine, video_id)
    assert after.progress == pytest.approx(0.4)
    assert after.updated_at > before


def test_heartbeat_clamps_progress(db: Session, engine: Engine) -> None:
    video_id = make_video(db)
    video = claim(engine)
    assert video and video.claim_id
    with Session(engine) as session:
        crud.heartbeat_video(
            session=session, video_id=video_id, claim_id=video.claim_id, progress=7.0
        )
    assert fetch(engine, video_id).progress == 1.0


def test_finish_replaces_old_rallies(db: Session, engine: Engine) -> None:
    video_id = make_video(db)
    db.add_all(
        [
            Rally(video_id=video_id, start=1, end=2),
            Rally(video_id=video_id, start=50, end=60),
        ]
    )
    db.commit()
    video = claim(engine)
    assert video and video.claim_id

    with Session(engine) as session:
        assert crud.finish_video(
            session=session,
            video_id=video_id,
            claim_id=video.claim_id,
            rallies=[(10.5, 20.0), (3.0, 7.25)],
        )

    done = fetch(engine, video_id)
    assert (done.status, done.progress, done.claim_id) == ("done", 1.0, None)
    assert rallies_of(engine, video_id) == [(3.0, 7.25), (10.5, 20.0)]


def test_finish_with_no_rallies(db: Session, engine: Engine) -> None:
    video_id = make_video(db)
    video = claim(engine)
    assert video and video.claim_id
    with Session(engine) as session:
        assert crud.finish_video(
            session=session, video_id=video_id, claim_id=video.claim_id, rallies=[]
        )
    assert fetch(engine, video_id).status == "done"
    assert rallies_of(engine, video_id) == []


def test_fail_records_the_error(db: Session, engine: Engine) -> None:
    video_id = make_video(db)
    video = claim(engine)
    assert video and video.claim_id
    with Session(engine) as session:
        assert crud.fail_video(
            session=session,
            video_id=video_id,
            claim_id=video.claim_id,
            reason=FailureReason.VIDEO_UNREADABLE,
            error="RuntimeError: " + "x" * 5000,
        )
    failed = fetch(engine, video_id)
    assert failed.status == "failed"
    assert failed.failure_reason == "Couldn't read this video"
    assert failed.error and failed.error.startswith("RuntimeError: x")
    assert len(failed.error) == crud.MAX_ERROR_LENGTH
    assert failed.claim_id is None


def test_release_puts_job_back_without_using_an_attempt(
    db: Session, engine: Engine
) -> None:
    video_id = make_video(db)
    video = claim(engine)
    assert video and video.claim_id
    with Session(engine) as session:
        assert crud.release_video(
            session=session, video_id=video_id, claim_id=video.claim_id
        )
    released = fetch(engine, video_id)
    assert (released.status, released.attempts, released.claim_id) == (
        "queued",
        0,
        None,
    )


def test_old_claim_cannot_write_after_job_was_taken_over(
    db: Session, engine: Engine
) -> None:
    """Worker A goes quiet (laptop asleep), its job is re-queued and taken by
    worker B. When A wakes up, none of its writes may land."""
    video_id = make_video(db)
    a = claim(engine, "worker-a")
    assert a and a.claim_id
    set_age(db, video_id, timedelta(minutes=10))
    with Session(engine) as session:
        assert crud.requeue_stale_videos(
            session=session, stale_after=timedelta(minutes=2)
        ) == (1, 0)
    b = claim(engine, "worker-b")
    assert b and b.claim_id and b.id == video_id

    with Session(engine) as session:
        kw = {"session": session, "video_id": video_id, "claim_id": a.claim_id}
        assert not crud.heartbeat_video(**kw, progress=0.9)
        assert not crud.finish_video(**kw, rallies=[(1.0, 2.0)])
        assert not crud.fail_video(**kw, reason=FailureReason.UNKNOWN, error="late")
        assert not crud.release_video(**kw)

    now = fetch(engine, video_id)
    assert (now.status, now.claim_id, now.claimed_by) == (
        "processing",
        b.claim_id,
        "worker-b",
    )
    assert now.error is None
    assert rallies_of(engine, video_id) == []


def test_writes_for_a_deleted_video_are_ignored(db: Session, engine: Engine) -> None:
    video_id = make_video(db)
    video = claim(engine)
    assert video and video.claim_id
    db.exec(delete(Video).where(col(Video.id) == video_id))
    db.commit()
    with Session(engine) as session:
        assert not crud.finish_video(
            session=session,
            video_id=video_id,
            claim_id=video.claim_id,
            rallies=[(1, 2)],
        )
    assert rallies_of(engine, video_id) == []


# --- stale jobs -------------------------------------------------------------


def test_requeue_stale_videos(db: Session, engine: Engine) -> None:
    stale = make_video(db, status="processing", age=timedelta(minutes=10), attempts=1)
    fresh = make_video(db, status="processing", age=timedelta(seconds=10), attempts=1)
    used_up = make_video(db, status="processing", age=timedelta(minutes=10), attempts=3)
    old_queued = make_video(db, status="queued", age=timedelta(hours=1))

    with Session(engine) as session:
        result = crud.requeue_stale_videos(
            session=session, stale_after=timedelta(minutes=2), max_attempts=3
        )
    assert result == (1, 1)

    assert fetch(engine, stale).status == "queued"
    assert fetch(engine, stale).claim_id is None
    assert fetch(engine, fresh).status == "processing"
    gave_up = fetch(engine, used_up)
    assert gave_up.status == "failed"
    assert gave_up.error and "stopped responding 3 times" in gave_up.error
    assert gave_up.failure_reason == FailureReason.KEPT_STOPPING
    assert fetch(engine, old_queued).status == "queued"  # queued isn't "stale"


def test_user_requeue_resets_attempts_and_failure(db: Session, engine: Engine) -> None:
    video_id = make_video(db, status="failed", attempts=3)
    db.exec(
        update(Video)
        .where(col(Video.id) == video_id)
        .values(failure_reason=FailureReason.UNKNOWN, error="boom")
    )
    db.commit()
    with Session(engine) as session:
        assert crud.queue_video(session=session, video_id=video_id) == "queued"
    queued = fetch(engine, video_id)
    assert (queued.status, queued.attempts) == ("queued", 0)
    assert (queued.failure_reason, queued.error) == (None, None)


# --- deleting vs. claiming ------------------------------------------------------


def test_delete_waits_for_a_claim_in_progress_then_refuses(
    db: Session, engine: Engine
) -> None:
    """A worker is mid-claim (row locked, status set, not committed yet) when the
    user deletes. The delete waits for the lock, then sees 'processing' and
    deletes nothing."""
    video_id = make_video(db)
    result: list[bool] = []

    def user_deletes() -> None:
        with Session(engine) as session:
            result.append(crud.delete_video(session=session, video_id=video_id))

    with Session(engine) as worker:
        worker.exec(
            select(Video.id).where(col(Video.id) == video_id).with_for_update()
        ).one()
        worker.exec(
            update(Video).where(col(Video.id) == video_id).values(status="processing")
        )
        deleting = threading.Thread(target=user_deletes)
        deleting.start()
        time.sleep(0.3)
        assert deleting.is_alive()  # waiting for the worker's row lock
        worker.commit()
    deleting.join(timeout=5)

    assert result == [False]
    assert fetch(engine, video_id).status == "processing"


def test_claim_skips_a_video_being_deleted(db: Session, engine: Engine) -> None:
    """The user's delete has locked the row (not committed yet): a worker skips it
    instead of waiting, and the video is gone afterwards."""
    video_id = make_video(db)
    with Session(engine) as user:
        user.exec(delete(Video).where(col(Video.id) == video_id))  # row now locked
        assert claim(engine) is None
        user.commit()
    assert claim(engine) is None
    with Session(engine) as session:
        assert session.get(Video, video_id) is None


def test_delete_video_crud(db: Session, engine: Engine) -> None:
    queued = make_video(db)
    processing = make_video(db, status="processing")
    with Session(engine) as session:
        assert crud.delete_video(session=session, video_id=queued)
        assert not crud.delete_video(session=session, video_id=processing)
        assert not crud.delete_video(session=session, video_id=uuid.uuid4())


# --- videos in progress per user -------------------------------------------------


def test_limit_holds_when_many_queue_at_once(db: Session, engine: Engine) -> None:
    """One user clicks "process" on 6 videos at the same moment, limit 2: exactly
    2 get queued. (Without the lock on the user row, each request could count 0-1
    active videos and let more through.)"""
    video_ids = [make_video(db, status="uploaded") for _ in range(6)]
    start_together = threading.Barrier(len(video_ids))
    results: list[str] = []
    errors: list[BaseException] = []

    def click(video_id: uuid.UUID) -> None:
        try:
            with Session(engine) as session:
                start_together.wait()
                results.append(
                    crud.queue_video(session=session, video_id=video_id, max_active=2)
                )
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=click, args=(v,)) for v in video_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    assert sorted(results) == ["limit_reached"] * 4 + ["queued"] * 2


def test_queue_video_results(db: Session, engine: Engine) -> None:
    queued = make_video(db, status="queued")
    fresh = make_video(db, status="uploaded")
    with Session(engine) as session:
        # already queued, at the limit: "not_allowed" (the more useful message)
        assert crud.queue_video(session=session, video_id=queued, max_active=1) == (
            "not_allowed"
        )
        assert crud.queue_video(session=session, video_id=fresh, max_active=1) == (
            "limit_reached"
        )
        assert (
            crud.queue_video(session=session, video_id=fresh, max_active=2) == "queued"
        )
        assert crud.queue_video(
            session=session, video_id=uuid.uuid4(), max_active=2
        ) == ("not_allowed")
