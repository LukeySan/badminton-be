import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from pwdlib.hashers.bcrypt import BcryptHasher

from app.core.config import settings

password_hash = PasswordHash(
    (
        Argon2Hasher(),
        BcryptHasher(),
    )
)


ALGORITHM = "HS256"
FILE_TOKEN_SCOPE = "video-file"  # a video's file and thumbnails
EXPORT_FILE_SCOPE = "export-file"  # one exported reel's MP4


def create_file_token(
    subject_id: uuid.UUID,
    scope: str = FILE_TOKEN_SCOPE,
    not_after: datetime | None = None,
) -> tuple[str, datetime]:
    """Short-lived token that allows fetching one video's (or export's) files.

    A <video src="..."> can't send the Authorization header, so the frontend gets
    a URL with this token instead of putting the (8-day) login token in a URL.
    Tokens with a scope are never accepted as login tokens (deps.get_current_user).
    not_after: expire no later than this (e.g. when an export's file is deleted).
    """
    expire = datetime.now(UTC) + timedelta(minutes=settings.FILE_TOKEN_EXPIRE_MINUTES)
    if not_after is not None:
        expire = min(expire, not_after)
    to_encode = {"exp": expire, "sub": str(subject_id), "scope": scope}
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM), expire


def verify_file_token(
    token: str, subject_id: uuid.UUID, scope: str = FILE_TOKEN_SCOPE
) -> bool:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.InvalidTokenError:
        return False
    return bool(payload.get("scope") == scope and payload.get("sub") == str(subject_id))


def create_access_token(subject: str | Any, expires_delta: timedelta) -> str:
    expire = datetime.now(UTC) + expires_delta
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_password(
    plain_password: str, hashed_password: str
) -> tuple[bool, str | None]:
    return password_hash.verify_and_update(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return password_hash.hash(password)
