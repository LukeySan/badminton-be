import uuid
from typing import Any

from sqlalchemy import update
from sqlmodel import Session, col, select

from app.core.security import get_password_hash, verify_password
from app.models import User, UserCreate, UserUpdate, Video, get_datetime_utc


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
            status="queued", progress=0.0, error=None, updated_at=get_datetime_utc()
        )
    )
    session.commit()
    return result.rowcount == 1
