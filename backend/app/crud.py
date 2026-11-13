import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, Literal

from sqlalchemy import extract, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlmodel import Session, col, delete, func, select

from app.core.security import get_password_hash, verify_password
from app.models import (
    MIN_RALLY_SECONDS,
    TRIM_TOLERANCE,
    Correction,
    CorrectionAction,
    FailureReason,
    Rally,
    RallyChange,
    RallyPublic,
    Reel,
    ReelEdit,
    ReelImpact,
    User,
    UserCreate,
    UserUpdate,
    Video,
    VideoPublic,
    get_datetime_utc,
)
from app.reel_ops import Retarget, Target, best_overlap, retarget_edit


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


ACTIVE_STATUSES = ("queued", "processing")
QueueResult = Literal["queued", "not_allowed", "limit_reached"]


def queue_video(
    *, session: Session, video_id: uuid.UUID, max_active: int | None = None
) -> QueueResult:
    """Queue a video for processing.

    "not_allowed": wrong status, no court yet, or no such video.
    "limit_reached": the owner already has max_active videos queued or processing.
    The owner's user row is locked while counting, so two "process" clicks at the
    same moment (two tabs, two videos) can't both see a count below the limit.
    """
    can_queue = (
        col(Video.id) == video_id,
        col(Video.status).in_(CAN_QUEUE_FROM),
        col(Video.court_corners).is_not(None),
    )
    if max_active is not None:
        owner_id = session.exec(
            select(Video.owner_id).where(col(Video.id) == video_id)
        ).first()
        if owner_id is None:
            return "not_allowed"
        session.exec(
            select(User.id).where(col(User.id) == owner_id).with_for_update()
        ).one()
        active = session.exec(
            select(func.count())
            .select_from(Video)
            .where(
                col(Video.owner_id) == owner_id,
                col(Video.status).in_(ACTIVE_STATUSES),
            )
        ).one()
        if active >= max_active:
            # "Already queued" is the more useful answer when both apply
            queueable = session.exec(select(Video.id).where(*can_queue)).first()
            session.rollback()  # releases the lock
            return "limit_reached" if queueable else "not_allowed"
    result = session.exec(
        update(Video)
        .where(*can_queue)
        .values(
            status="queued",
            progress=0.0,
            failure_reason=None,
            error=None,
            attempts=0,
            updated_at=get_datetime_utc(),
        )
    )
    session.commit()
    return "queued" if result.rowcount == 1 else "not_allowed"


def delete_video(*, session: Session, video_id: uuid.UUID) -> bool:
    """Delete the video unless a worker is processing it. Its rallies and corrections
    go with it (database cascade); the caller removes the files.

    One conditional DELETE: if a worker claims the video at the same moment, the row
    lock decides who goes first, so a video is never deleted mid-processing.
    Returns False if nothing was deleted (processing, or already gone)."""
    result = session.exec(
        delete(Video).where(
            col(Video.id) == video_id, col(Video.status) != "processing"
        )
    )
    session.commit()
    return result.rowcount == 1


# --- Video lists with rally stats ---------------------------------------------
# rally_count / play_seconds come from one GROUP BY over the rallies, joined onto
# the video query, so a list of N videos is still one query, not N + 1.


def _rally_stats() -> Any:
    return (
        select(
            col(Rally.video_id).label("video_id"),
            func.count().label("rally_count"),
            func.sum(col(Rally.end) - col(Rally.start)).label("play_seconds"),
        )
        .group_by(col(Rally.video_id))
        .subquery()
    )


def _to_public(
    video: Video, rally_count: int | None, play_seconds: float | None
) -> VideoPublic:
    return VideoPublic.model_validate(
        video,
        update={
            "rally_count": rally_count or 0,
            "play_seconds": round(play_seconds or 0.0, 2),
        },
    )


def list_videos(
    *, session: Session, owner_id: uuid.UUID | None, skip: int = 0, limit: int = 100
) -> tuple[list[VideoPublic], int]:
    """Videos newest first, with rally stats, and the total count.
    owner_id=None means everyone's videos."""
    stats = _rally_stats()
    statement = select(Video, stats.c.rally_count, stats.c.play_seconds).outerjoin(
        stats, col(Video.id) == stats.c.video_id
    )
    count_statement = select(func.count()).select_from(Video)
    if owner_id is not None:
        statement = statement.where(col(Video.owner_id) == owner_id)
        count_statement = count_statement.where(col(Video.owner_id) == owner_id)
    rows = session.exec(
        statement.order_by(col(Video.created_at).desc(), col(Video.id))
        .offset(skip)
        .limit(limit)
    ).all()
    count = session.exec(count_statement).one()
    return [_to_public(video, n, s) for video, n, s in rows], count


def get_video_public(*, session: Session, video: Video) -> VideoPublic:
    """One video as the API returns it, with its rally stats."""
    rally_count, play_seconds = session.exec(
        select(
            func.count(col(Rally.id)),
            func.sum(col(Rally.end) - col(Rally.start)),
        ).where(col(Rally.video_id) == video.id)
    ).one()
    return _to_public(video, rally_count, play_seconds)


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
            failure_reason=None,
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
    # The detector's bounds are kept separately, for when the user corrects them
    new_rallies = [
        Rally(video_id=video_id, start=start, end=end, source=source,
              detected_start=start, detected_end=end)
        for start, end in rallies
    ]  # fmt: skip
    session.add_all(new_rallies)
    session.flush()  # assigns the new rally ids
    # New ids: point reel clips at the new rally covering the same time
    targets = [Target(r.id, r.start, r.end) for r in new_rallies if r.id is not None]
    _retarget_reels(session, video_id, lambda clip: best_overlap(clip, targets))
    session.commit()
    return True


def _retarget_reels(
    session: Session,
    video_id: uuid.UUID,
    retarget: Retarget,
    join: tuple[int, int] | None = None,
) -> list[ReelImpact]:
    """Adjust the video's reels to changed rallies (see app/reel_ops.py), in the
    caller's transaction. Changed reels get a new updated_at, so an editor that has
    one open gets a 409 on its next save and reloads.
    FOR UPDATE: a user's save at the same moment waits, then gets that 409, instead
    of overwriting this or being overwritten."""
    reels = session.exec(
        select(Reel)
        .where(col(Reel.video_id) == video_id, col(Reel.deleted_at).is_(None))
        .order_by(col(Reel.updated_at).desc())
        .with_for_update()
    ).all()
    impacts = []
    for reel in reels:
        old = ReelEdit.model_validate(reel.edit)
        new, impact = retarget_edit(old, retarget, join)
        if new == old:
            continue
        reel.edit = new.model_dump(mode="json")  # a new dict: SQLAlchemy sees it
        reel.updated_at = get_datetime_utc()
        session.add(reel)
        impacts.append(
            ReelImpact(
                id=reel.id,
                title=reel.title,
                updated_at=reel.updated_at,
                clips_removed=impact.clips_removed,
                pieces_trimmed=impact.pieces_trimmed,
                pieces_removed=impact.pieces_removed,
            )
        )
    return impacts


def fail_video(
    *,
    session: Session,
    video_id: uuid.UUID,
    claim_id: uuid.UUID,
    reason: FailureReason,
    error: str,
) -> bool:
    """Mark the video failed: `reason` is shown to the user, `error` (technical
    details) stays internal. False = no longer ours."""
    result = session.exec(
        update(Video)
        .where(*_claimed(video_id, claim_id))
        .values(
            status="failed",
            failure_reason=reason,
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
            failure_reason=FailureReason.KEPT_STOPPING,
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


def queue_stats(*, session: Session) -> tuple[int, int, float | None]:
    """(queued, processing, seconds the oldest queued video has waited) in one
    query. Uses the database clock, like the heartbeats."""
    queued = col(Video.status) == "queued"
    n_queued, n_processing, oldest = session.exec(
        select(
            func.count().filter(queued),
            func.count().filter(col(Video.status) == "processing"),
            extract(
                "epoch", func.now() - func.min(col(Video.updated_at)).filter(queued)
            ),
        )
    ).one()
    return n_queued, n_processing, None if oldest is None else float(oldest)


# --- Reels --------------------------------------------------------------------
# Soft delete: deleted_at is set and the reel disappears from the API, but the row
# stays (undo, support, and what people delete is useful to know later).


def _live_reel(reel_id: uuid.UUID) -> tuple[Any, ...]:
    return (col(Reel.id) == reel_id, col(Reel.deleted_at).is_(None))


def lock_video_for_reel_save(*, session: Session, video_id: uuid.UUID) -> None:
    """FOR SHARE on the video row until the caller's commit: rally corrections and
    re-analysis (which lock or update that row) wait until the reel is saved, so a
    reel can't be validated against rallies that change before it's written."""
    session.exec(
        select(Video.id).where(col(Video.id) == video_id).with_for_update(read=True)
    ).one()


def video_rally_bounds(
    *, session: Session, video_id: uuid.UUID
) -> dict[int, tuple[float, float]]:
    rows = session.exec(
        select(Rally.id, Rally.start, Rally.end).where(col(Rally.video_id) == video_id)
    ).all()
    return {i: (start, end) for i, start, end in rows if i is not None}


def list_reels(
    *, session: Session, video_id: uuid.UUID, skip: int = 0, limit: int = 100
) -> tuple[Sequence[Reel], int]:
    """A video's reels (not deleted), most recently saved first, and the total."""
    live = (col(Reel.video_id) == video_id, col(Reel.deleted_at).is_(None))
    reels = session.exec(
        select(Reel)
        .where(*live)
        .order_by(col(Reel.updated_at).desc(), col(Reel.id))
        .offset(skip)
        .limit(limit)
    ).all()
    count = session.exec(select(func.count()).select_from(Reel).where(*live)).one()
    return reels, count


def create_reel(
    *, session: Session, video: Video, title: str, edit: ReelEdit
) -> Reel | None:
    """Create the video's reel; None if it already has one (one per video).
    INSERT ... ON CONFLICT DO NOTHING on the partial unique index: of two creates at
    the same moment exactly one inserts, the other gets None (no error, no retry)."""
    now = get_datetime_utc()
    reel_id = session.exec(
        pg_insert(Reel)
        .values(
            id=uuid.uuid4(),
            video_id=video.id,
            owner_id=video.owner_id,
            title=title,
            edit=edit.without_empty_clips().model_dump(mode="json"),
            created_at=now,
            updated_at=now,
        )
        .on_conflict_do_nothing(
            index_elements=[col(Reel.video_id)],
            index_where=col(Reel.deleted_at).is_(None),
        )
        .returning(col(Reel.id))
    ).first()
    session.commit()
    return None if reel_id is None else session.get(Reel, reel_id[0])


def get_video_reel(*, session: Session, video_id: uuid.UUID) -> Reel | None:
    """The video's reel (not deleted), or None."""
    return session.exec(
        select(Reel).where(
            col(Reel.video_id) == video_id, col(Reel.deleted_at).is_(None)
        )
    ).first()


def save_video_reel(
    *,
    session: Session,
    video: Video,
    title: str,
    edit: ReelEdit,
    expected_updated_at: datetime | None,
) -> tuple[Reel | None, bool]:
    """Create (expected_updated_at None) or update the video's reel, only if it is
    as the caller last saw it: none yet, or unchanged since it was loaded.
    Returns (reel, created); reel None = it changed meanwhile (stale save)."""
    if expected_updated_at is None:
        return create_reel(session=session, video=video, title=title, edit=edit), True
    reel = get_video_reel(session=session, video_id=video.id)
    if reel is None or not update_reel(
        session=session,
        reel_id=reel.id,
        title=title,
        edit=edit,
        expected_updated_at=expected_updated_at,
    ):
        return None, False
    session.refresh(reel)
    return reel, False


def get_reel(*, session: Session, reel_id: uuid.UUID) -> Reel | None:
    """The reel, or None if it doesn't exist or was deleted."""
    return session.exec(select(Reel).where(*_live_reel(reel_id))).first()


def update_reel(
    *,
    session: Session,
    reel_id: uuid.UUID,
    title: str,
    edit: ReelEdit,
    expected_updated_at: datetime,
) -> bool:
    """Replace the title and edit list, only if the reel is unchanged since the
    caller loaded it (updated_at still matches; optimistic locking). One
    conditional UPDATE, so two saves at once can't both pass the check.
    False = changed since, or deleted."""
    result = session.exec(
        update(Reel)
        .where(*_live_reel(reel_id), col(Reel.updated_at) == expected_updated_at)
        .values(
            title=title,
            edit=edit.without_empty_clips().model_dump(mode="json"),
            updated_at=get_datetime_utc(),
        )
    )
    session.commit()
    return result.rowcount == 1


def soft_delete_reel(*, session: Session, reel_id: uuid.UUID) -> bool:
    """False if the reel doesn't exist or was already deleted."""
    result = session.exec(
        update(Reel).where(*_live_reel(reel_id)).values(deleted_at=get_datetime_utc())
    )
    session.commit()
    return result.rowcount == 1


# --- Rally corrections ----------------------------------------------------------
# Each correction is one transaction: lock the video row (so it can't be queued or
# re-analyzed meanwhile, and reel saves wait), change the rallies, log the change in
# `correction`, adjust the reels, commit. Refusals raise CorrectionRefused.


class CorrectionRefused(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code, self.detail = status_code, detail


def numbered_rallies(*, session: Session, video_id: uuid.UUID) -> list[RallyPublic]:
    rallies = session.exec(
        select(Rally)
        .where(col(Rally.video_id) == video_id)
        .order_by(col(Rally.start), col(Rally.id))
    ).all()
    return [
        RallyPublic(id=r.id, number=n, start=r.start, end=r.end)
        for n, r in enumerate(rallies, 1)
        if r.id is not None
    ]


class _Correction:
    """One correction in progress (see the functions below)."""

    def __init__(self, session: Session, video_id: uuid.UUID, user_id: uuid.UUID):
        self.session, self.video_id, self.user_id = session, video_id, user_id
        self.operation_id = uuid.uuid4()
        video = session.exec(
            select(Video).where(col(Video.id) == video_id).with_for_update()
        ).first()
        if video is None:  # deleted by another request meanwhile
            session.rollback()
            raise CorrectionRefused(404, "Video not found")
        if video.status in ACTIVE_STATUSES:
            session.rollback()
            raise CorrectionRefused(
                409, f"Can't change rallies while the video is {video.status}"
            )
        self.duration = video.duration
        self.rallies = list(
            session.exec(
                select(Rally)
                .where(col(Rally.video_id) == video_id)
                .order_by(col(Rally.start), col(Rally.id))
            ).all()
        )

    def refuse(self, status_code: int, detail: str) -> CorrectionRefused:
        self.session.rollback()
        return CorrectionRefused(status_code, detail)

    def number(self, rally: Rally) -> int:
        return self.rallies.index(rally) + 1

    def get(self, rally_id: int) -> Rally:
        for rally in self.rallies:
            if rally.id == rally_id:
                return rally
        raise self.refuse(404, "Rally not found")

    def check_times(
        self, start: float, end: float, ignore: Sequence[Rally] = ()
    ) -> None:
        if self.duration is not None and end > self.duration + TRIM_TOLERANCE:
            raise self.refuse(
                422, f"end must be within the video (0-{self.duration:.2f} s)"
            )
        for other in self.rallies:
            if other not in ignore and start < other.end and other.start < end:
                raise self.refuse(
                    409, f"Overlaps rally {self.number(other)} "
                    f"({other.start:.2f}-{other.end:.2f} s)",
                )  # fmt: skip

    def log(
        self,
        action: CorrectionAction,
        before: Rally | None,
        after: Rally | None,
        detected: tuple[float | None, float | None],
    ) -> None:
        self.session.add(
            Correction(
                video_id=self.video_id,
                operation_id=self.operation_id,
                user_id=self.user_id,
                action=action,
                rally_id=before.id if before else None,
                result_rally_id=after.id if after else None,
                original_start=detected[0],
                original_end=detected[1],
                edited_start=after.start if after else None,
                edited_end=after.end if after else None,
            )
        )

    def finish(
        self,
        *,
        added: Sequence[Rally] = (),
        updated: Sequence[Rally] = (),
        deleted: Sequence[int] = (),
        retarget: Retarget | None = None,
        join: tuple[int, int] | None = None,
    ) -> RallyChange:
        reels = (
            _retarget_reels(self.session, self.video_id, retarget, join)
            if retarget
            else []
        )
        self.session.commit()
        rallies = numbered_rallies(session=self.session, video_id=self.video_id)
        by_id = {r.id: r for r in rallies}
        return RallyChange(
            added=[by_id[r.id] for r in added if r.id in by_id],
            updated=[by_id[r.id] for r in updated if r.id in by_id],
            deleted=list(deleted),
            rallies=rallies,
            reels=reels,
        )


def _detected(rally: Rally) -> tuple[float | None, float | None]:
    return rally.detected_start, rally.detected_end


def add_rally(
    *,
    session: Session,
    video_id: uuid.UUID,
    user_id: uuid.UUID,
    start: float,
    end: float,
) -> RallyChange:
    c = _Correction(session, video_id, user_id)
    c.check_times(start, end)
    rally = Rally(video_id=video_id, start=start, end=end, source="user")
    session.add(rally)
    session.flush()
    c.log("add", None, rally, (None, None))
    return c.finish(added=[rally])  # no reel points at a new rally


def edit_rally(
    *,
    session: Session,
    video_id: uuid.UUID,
    user_id: uuid.UUID,
    rally_id: int,
    start: float,
    end: float,
) -> RallyChange:
    c = _Correction(session, video_id, user_id)
    rally = c.get(rally_id)
    c.check_times(start, end, ignore=[rally])
    if (rally.start, rally.end) == (start, end):
        return c.finish()
    rally.start, rally.end = start, end
    session.add(rally)
    c.log("edit", rally, rally, _detected(rally))
    target = Target(rally_id, start, end)
    return c.finish(
        updated=[rally],
        retarget=lambda clip: [target] if clip.rally_id == rally_id else None,
    )


def delete_rally(
    *, session: Session, video_id: uuid.UUID, user_id: uuid.UUID, rally_id: int
) -> RallyChange:
    c = _Correction(session, video_id, user_id)
    rally = c.get(rally_id)
    c.log("delete", rally, None, _detected(rally))
    session.delete(rally)
    return c.finish(
        deleted=[rally_id],
        retarget=lambda clip: [] if clip.rally_id == rally_id else None,
    )


def split_rally(
    *,
    session: Session,
    video_id: uuid.UUID,
    user_id: uuid.UUID,
    rally_id: int,
    at: float,
) -> RallyChange:
    """The first part keeps the rally's id, the second part is a new rally. Both
    remember the detector's bounds of the rally they came from."""
    c = _Correction(session, video_id, user_id)
    rally = c.get(rally_id)
    if not (rally.start + MIN_RALLY_SECONDS <= at <= rally.end - MIN_RALLY_SECONDS):
        raise c.refuse(
            422, f"Split point must be inside rally {c.number(rally)} "
            f"({rally.start:.2f}-{rally.end:.2f} s)",
        )  # fmt: skip
    second = Rally(
        video_id=video_id,
        start=at,
        end=rally.end,
        source=rally.source,
        detected_start=rally.detected_start,
        detected_end=rally.detected_end,
    )
    rally.end = at
    session.add_all([rally, second])
    session.flush()
    assert second.id is not None
    c.log("split", rally, rally, _detected(rally))
    c.log("split", rally, second, _detected(rally))
    targets = [Target(rally_id, rally.start, at), Target(second.id, at, second.end)]
    return c.finish(
        added=[second],
        updated=[rally],
        retarget=lambda clip: targets if clip.rally_id == rally_id else None,
    )


def merge_rallies(
    *,
    session: Session,
    video_id: uuid.UUID,
    user_id: uuid.UUID,
    rally_ids: Sequence[int],
) -> RallyChange:
    """Two neighbouring rallies become one new rally from the first's start to the
    second's end (a new id: neither old rally survives as it was)."""
    c = _Correction(session, video_id, user_id)
    if len(set(rally_ids)) != 2:
        raise c.refuse(422, "Give two different rallies")
    first, second = sorted((c.get(i) for i in rally_ids), key=c.number)
    if c.number(second) != c.number(first) + 1:
        raise c.refuse(
            409, f"Rallies {c.number(first)} and {c.number(second)} aren't next to "
            "each other; only neighbouring rallies can be merged",
        )  # fmt: skip
    merged = Rally(
        video_id=video_id,
        start=first.start,
        end=second.end,
        source=first.source,
        # The span of the detector's rallies it came from (one side may be the user's)
        detected_start=first.detected_start if first.detected_start is not None
        else second.detected_start,
        detected_end=second.detected_end if second.detected_end is not None
        else first.detected_end,
    )  # fmt: skip
    session.add(merged)
    session.flush()
    assert merged.id is not None and first.id is not None and second.id is not None
    c.log("merge", first, merged, _detected(first))
    c.log("merge", second, merged, _detected(second))
    old_ids = {first.id, second.id}
    session.delete(first)
    session.delete(second)
    target = Target(merged.id, merged.start, merged.end)
    return c.finish(
        added=[merged],
        deleted=sorted(old_ids),
        retarget=lambda clip: [target] if clip.rally_id in old_ids else None,
        join=(first.id, second.id),
    )
