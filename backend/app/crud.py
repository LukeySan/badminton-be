import uuid
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from sqlalchemy import update
from sqlmodel import Session, col, delete, func, select

from app.core.security import get_password_hash, verify_password
from app.models import Rally, User, UserCreate, UserUpdate, Video, get_datetime_utc


def create_user(*, session: Session, user_create: UserCreate) -> User:
    db_obj = User.model_validate(
        user_create, update={"hashed_password": get_password_hash(user_create.password)}
    )
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj


def update_user(*, session: Session, db_user: User, user_in: UserUpdate) -> Any:
    user_data = user_in.model_dump(exclude_unset=True)
    extra_data = {}
    if "password" in user_data:
        password = user_data["password"]
        hashed_password = get_password_hash(password)
        extra_data["hashed_password"] = hashed_password
    db_user.sqlmodel_update(user_data, update=extra_data)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


def get_user_by_email(*, session: Session, email: str) -> User | None:
    statement = select(User).where(User.email == email)
    session_user = session.exec(statement).first()
    return session_user


# Dummy hash to use for timing attack prevention when user is not found
# This is an Argon2 hash of a random password, used to ensure constant-time comparison
DUMMY_HASH = "$argon2id$v=19$m=65536,t=3,p=4$MjQyZWE1MzBjYjJlZTI0Yw$YTU4NGM5ZTZmYjE2NzZlZjY0ZWY3ZGRkY2U2OWFjNjk"


def authenticate(*, session: Session, email: str, password: str) -> User | None:
    db_user = get_user_by_email(session=session, email=email)
    if not db_user:
        # Prevent timing attacks by running password verification even when user doesn't exist
        # This ensures the response time is similar whether or not the email exists
        verify_password(password, DUMMY_HASH)
        return None
    verified, updated_password_hash = verify_password(password, db_user.hashed_password)
    if not verified:
        return None
    if updated_password_hash:
        db_user.hashed_password = updated_password_hash
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
    return db_user


# Video status changes are single conditional UPDATEs (update(...).where(status in
# ...)), not "load the row, check it in Python, save it": that leaves a gap between
# check and write where two clicks on "process", or two workers claiming a job,
# could both succeed.

# done/failed can be queued again: retry, or reprocess after fixing the court
CAN_QUEUE_FROM = ("uploaded", "done", "failed")
CAN_EDIT_COURT_IN = ("uploaded", "done", "failed")


def set_video_court(
    *, session: Session, video_id: uuid.UUID, corners: list[list[float]]
) -> bool:
    """Returns False if the video is queued/processing (or doesn't exist)."""
    result = session.exec(
        update(Video)
        .where(
            col(Video.id) == video_id,
            col(Video.status).in_(CAN_EDIT_COURT_IN),
        )
        .values(court_corners=corners, updated_at=get_datetime_utc())
    )
    session.commit()
    return result.rowcount == 1


def queue_video(*, session: Session, video_id: uuid.UUID) -> bool:
    """Returns False if the video can't be queued right now (or doesn't exist)."""
    result = session.exec(
        update(Video)
        .where(
            col(Video.id) == video_id,
            col(Video.status).in_(CAN_QUEUE_FROM),
            col(Video.court_corners).is_not(None),
        )
        .values(
            status="queued",
            progress=0.0,
            error=None,
            attempts=0,
            updated_at=get_datetime_utc(),
        )
    )
    session.commit()
    return result.rowcount == 1


# --- Worker jobs ------------------------------------------------------------
# Workers may run on several machines at once (Mac, 2070 PC, laptop). Heartbeat
# times use the database's clock (func.now()), not each machine's, so clock
# differences between machines can't make a healthy job look stale.

MAX_ATTEMPTS = 3
MAX_ERROR_LENGTH = 2000


def _claimed(video_id: uuid.UUID, claim_id: uuid.UUID) -> tuple[Any, ...]:
    """WHERE clause for "this worker still owns this job"."""
    return (
        col(Video.id) == video_id,
        col(Video.status) == "processing",
        col(Video.claim_id) == claim_id,
    )


def claim_next_video(*, session: Session, claimed_by: str) -> Video | None:
    """Take the oldest queued video and mark it processing, or return None.

    FOR UPDATE locks the selected row until commit; SKIP LOCKED makes another
    worker claiming at the same moment skip that row and take the next video,
    instead of waiting for the lock and then finding nothing.
    """
    video_id = session.exec(
        select(Video.id)
        .where(col(Video.status) == "queued")
        # updated_at = when it was queued: nothing else changes a queued video
        .order_by(col(Video.updated_at), col(Video.created_at))
        .limit(1)
        .with_for_update(skip_locked=True)
    ).first()
    if video_id is None:
        session.rollback()  # end the transaction
        return None
    session.exec(
        update(Video)
        .where(col(Video.id) == video_id)
        .values(
            status="processing",
            progress=0.0,
            error=None,
            claim_id=uuid.uuid4(),
            claimed_by=claimed_by[:255],
            attempts=col(Video.attempts) + 1,
            updated_at=func.now(),
        )
    )
    session.commit()
    return session.get(Video, video_id)


def heartbeat_video(
    *, session: Session, video_id: uuid.UUID, claim_id: uuid.UUID, progress: float
) -> bool:
    """Refresh the heartbeat and progress. False = the job is no longer ours."""
    result = session.exec(
        update(Video)
        .where(*_claimed(video_id, claim_id))
        .values(progress=min(max(progress, 0.0), 1.0), updated_at=func.now())
    )
    session.commit()
    return result.rowcount == 1


def finish_video(
    *,
    session: Session,
    video_id: uuid.UUID,
    claim_id: uuid.UUID,
    rallies: Sequence[tuple[float, float]],
    source: str = "rules",
) -> bool:
    """Replace the video's rallies and mark it done, in one transaction, so the app
    never sees "done" with old or half-written rallies. False = no longer ours."""
    result = session.exec(
        update(Video)
        .where(*_claimed(video_id, claim_id))
        .values(status="done", progress=1.0, claim_id=None, updated_at=func.now())
    )
    if result.rowcount != 1:
        session.rollback()
        return False
    session.exec(delete(Rally).where(col(Rally.video_id) == video_id))
    session.add_all(
        Rally(video_id=video_id, start=start, end=end, source=source)
        for start, end in rallies
    )
    session.commit()
    return True


def fail_video(
    *, session: Session, video_id: uuid.UUID, claim_id: uuid.UUID, error: str
) -> bool:
    """Mark the video failed with an error message. False = no longer ours."""
    result = session.exec(
        update(Video)
        .where(*_claimed(video_id, claim_id))
        .values(
            status="failed",
            error=error[:MAX_ERROR_LENGTH],
            claim_id=None,
            updated_at=func.now(),
        )
    )
    session.commit()
    return result.rowcount == 1


def release_video(
    *, session: Session, video_id: uuid.UUID, claim_id: uuid.UUID
) -> bool:
    """Put a job back in the queue, e.g. when the worker is stopped with Ctrl+C.
    Doesn't count as an attempt."""
    result = session.exec(
        update(Video)
        .where(*_claimed(video_id, claim_id))
        .values(
            status="queued",
            progress=0.0,
            claim_id=None,
            attempts=col(Video.attempts) - 1,
            updated_at=func.now(),
        )
    )
    session.commit()
    return result.rowcount == 1


def requeue_stale_videos(
    *, session: Session, stale_after: timedelta, max_attempts: int = MAX_ATTEMPTS
) -> tuple[int, int]:
    """Rescue jobs whose worker stopped sending heartbeats (crashed, lost its
    connection, machine went to sleep). Re-queues them, or fails them once they have
    used up max_attempts, so a video that crashes the worker every time doesn't
    loop forever. Returns (requeued, failed)."""
    stale = (
        col(Video.status) == "processing",
        col(Video.updated_at) < func.now() - stale_after,
    )
    failed = session.exec(
        update(Video)
        .where(*stale, col(Video.attempts) >= max_attempts)
        .values(
            status="failed",
            error=(
                f"Processing stopped responding {max_attempts} times "
                "(the worker crashed or lost its connection)."
            ),
            claim_id=None,
            updated_at=func.now(),
        )
    )
    requeued = session.exec(
        update(Video)
        .where(*stale, col(Video.attempts) < max_attempts)
        .values(status="queued", progress=0.0, claim_id=None, updated_at=func.now())
    )
    session.commit()
    return requeued.rowcount, failed.rowcount
