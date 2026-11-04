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
FILE_TOKEN_SCOPE = "video-file"


def create_file_token(video_id: uuid.UUID) -> tuple[str, datetime]:
    """Short-lived token that allows streaming one video's file.

    A <video src="..."> can't send the Authorization header, so the frontend gets
    a URL with this token instead of putting the (8-day) login token in a URL.
    """
    expire = datetime.now(UTC) + timedelta(minutes=settings.FILE_TOKEN_EXPIRE_MINUTES)
    to_encode = {"exp": expire, "sub": str(video_id), "scope": FILE_TOKEN_SCOPE}
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM), expire


def verify_file_token(token: str, video_id: uuid.UUID) -> bool:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.InvalidTokenError:
        return False
    return bool(
        payload.get("scope") == FILE_TOKEN_SCOPE and payload.get("sub") == str(video_id)
    )


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
