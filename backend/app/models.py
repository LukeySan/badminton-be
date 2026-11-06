import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import EmailStr, computed_field, model_validator
from pydantic import Field as PydanticField
from sqlalchemy import Column, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel


def get_datetime_utc() -> datetime:
    return datetime.now(UTC)


# Shared properties
class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)


class UserRegister(SQLModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on update, all are optional
class UserUpdate(SQLModel):
    email: EmailStr | None = Field(default=None, max_length=255)
    is_active: bool | None = None
    is_superuser: bool | None = None
    full_name: str | None = Field(default=None, max_length=255)
    password: str | None = Field(default=None, min_length=8, max_length=128)


class UserUpdateMe(SQLModel):
    full_name: str | None = Field(default=None, max_length=255)
    email: EmailStr | None = Field(default=None, max_length=255)


class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


# Database model, database table inferred from class name
class User(UserBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    hashed_password: str
    created_at: datetime | None = Field(default_factory=get_datetime_utc)
    videos: list[Video] = Relationship(back_populates="owner", cascade_delete=True)


# Properties to return via API, id is always required
class UserPublic(UserBase):
    id: uuid.UUID
    created_at: datetime | None = None


class UsersPublic(SQLModel):
    data: list[UserPublic]
    count: int


# ---------------------------------------------------------------------------
# Videos and rallies
# ---------------------------------------------------------------------------

# uploaded -> queued -> processing -> done | failed  (done/failed can be queued again)
VideoStatus = Literal["uploaded", "queued", "processing", "done", "failed"]

# Court corners are fractions 0-1 of the displayed width/height (resolution
# independent); pixels only exist inside the worker.
Fraction01 = Annotated[float, PydanticField(ge=0, le=1)]
Point = tuple[Fraction01, Fraction01]


class CourtUpdate(SQLModel):
    corners: list[Point] = Field(
        min_length=4,
        max_length=4,
        description="Fractions 0-1 of width/height, order: far-left, far-right, near-right, near-left",
        schema_extra={
            "examples": [
                [[0.478, 0.347], [0.777, 0.369], [0.930, 0.943], [0.054, 0.781]]
            ]
        },
    )

    @model_validator(mode="after")
    def check_order(self) -> Self:
        # Wrong tap order is the likely user mistake, and it gives bad rallies without
        # any error later, so catch it here. With the camera behind the baseline the far
        # baseline is higher in the image (smaller y) than the near one.
        fl, fr, nr, nl = self.corners
        if max(fl[1], fr[1]) >= min(nr[1], nl[1]):
            raise ValueError(
                "far corners must be above the near corners; tap far-left, "
                "far-right, near-right, near-left"
            )
        if fl[0] >= fr[0] or nl[0] >= nr[0]:
            raise ValueError("left corners must be left of the right corners")
        return self


# Database model
class Video(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE", index=True
    )
    filename: str = Field(max_length=255)  # the user's original name, display only
    storage_path: str = Field(max_length=255)  # relative to settings.STORAGE_DIR
    status: str = Field(default="uploaded", max_length=20, index=True)
    progress: float = 0.0  # 0-1 while processing
    error: str | None = Field(default=None, sa_type=Text)
    duration: float | None = None
    fps: float | None = None
    width: int | None = None  # as displayed (rotation applied)
    height: int | None = None
    # [[x, y] x4]. none_as_null: Python None is stored as SQL NULL, not JSON 'null'
    court_corners: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSONB(none_as_null=True))
    )
    # Worker bookkeeping (not in the API). claim_id is new on every claim; a worker
    # only writes to the video while claim_id is still its own, so a worker that was
    # wrongly declared dead can't overwrite the results of the one that took over.
    claim_id: uuid.UUID | None = None
    claimed_by: str | None = Field(default=None, max_length=255)  # "hostname:pid"
    attempts: int = Field(default=0, sa_column_kwargs={"server_default": "0"})
    created_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),
    )
    # While processing, the worker refreshes this every few seconds as a heartbeat,
    # using the database's clock (see crud.claim_next_video)
    updated_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={"onupdate": get_datetime_utc},
    )
    owner: User | None = Relationship(back_populates="videos")
    rallies: list[Rally] = Relationship(
        back_populates="video",
        cascade_delete=True,
        sa_relationship_kwargs={"order_by": "Rally.start"},
    )


class VideoPublic(SQLModel):
    id: uuid.UUID
    owner_id: uuid.UUID
    filename: str
    status: VideoStatus
    progress: float
    error: str | None
    duration: float | None
    fps: float | None
    width: int | None
    height: int | None
    court_corners: list[Point] | None
    created_at: datetime
    updated_at: datetime


class VideosPublic(SQLModel):
    data: list[VideoPublic]
    count: int


class Rally(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    video_id: uuid.UUID = Field(
        foreign_key="video.id", nullable=False, ondelete="CASCADE", index=True
    )
    start: float  # seconds
    end: float
    source: str = Field(default="rules", max_length=10)  # 'rules' | 'model'
    created_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),
    )
    video: Video | None = Relationship(back_populates="rallies")


class RallyPublic(SQLModel):
    id: int
    start: float
    end: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


class RalliesPublic(SQLModel):
    data: list[RallyPublic]
    count: int


class Correction(SQLModel, table=True):
    """Not used yet: user edits to rallies, stored next to the detector's original
    as future training data."""

    id: int | None = Field(default=None, primary_key=True)
    rally_id: int | None = None
    video_id: uuid.UUID = Field(
        foreign_key="video.id", nullable=False, ondelete="CASCADE", index=True
    )
    original_start: float | None = None
    original_end: float | None = None
    edited_start: float | None = None
    edited_end: float | None = None
    action: str = Field(max_length=10)  # 'edit' | 'delete' | 'add'
    created_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),
    )


class FileUrl(SQLModel):
    url: str  # relative to the API host, includes a signed token
    expires_at: datetime


# Generic message
class Message(SQLModel):
    message: str


# JSON payload containing access token
class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"


# Contents of JWT token
class TokenPayload(SQLModel):
    sub: str | None = None


class NewPassword(SQLModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)
