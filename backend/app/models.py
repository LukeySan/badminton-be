import itertools
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    EmailStr,
    computed_field,
    model_validator,
)
from pydantic import Field as PydanticField
from sqlalchemy import Column, DateTime, Index, Text, text
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


class FailureReason(StrEnum):
    """Why a video failed, in words for the user (Video.failure_reason). The technical
    details go to Video.error, which stays internal (it can contain server paths).
    An enum so the frontend's generated types list every possible value."""

    VIDEO_UNREADABLE = "Couldn't read this video"
    COURT_NOT_VISIBLE = "Court not visible"
    KEPT_STOPPING = "Processing kept stopping on this video"
    UNKNOWN = "Something went wrong while processing this video"


# Court corners are fractions of the displayed width/height (resolution independent);
# pixels only exist inside the worker. 0-1 is inside the frame, but a corner may be
# a little outside it (phone on a tripod; the court-tapping screen shows a gray margin
# around the frame so it can be tapped there):
COURT_MARGIN_OK = 0.1  # up to 10% outside the frame: fine, no warning (e.g. match 5)
COURT_MARGIN_MAX = (
    0.3  # up to 30% outside: accepted, with the court_out_of_frame warning
)
# Further out is rejected: that's a mis-tap, not a real court.
CourtCoord = Annotated[
    float, PydanticField(ge=-COURT_MARGIN_MAX, le=1 + COURT_MARGIN_MAX)
]
Point = tuple[CourtCoord, CourtCoord]


def is_court_out_of_frame(corners: Sequence[Sequence[float]]) -> bool:
    """True if any corner is more than COURT_MARGIN_OK outside the video frame."""
    return any(
        v < -COURT_MARGIN_OK or v > 1 + COURT_MARGIN_OK
        for point in corners
        for v in point
    )


class CourtUpdate(SQLModel):
    corners: list[Point] = Field(
        min_length=4,
        max_length=4,
        description=(
            "Fractions of width/height, order: far-left, far-right, near-right, "
            "near-left. 0-1 is inside the frame; -0.3 to 1.3 is accepted (see "
            "court_out_of_frame on the video)."
        ),
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
    # When failed: a FailureReason for the user, and the technical details (internal)
    failure_reason: str | None = Field(default=None, max_length=255)
    error: str | None = Field(default=None, sa_type=Text)
    duration: float | None = None
    fps: float | None = None
    width: int | None = None  # as displayed (rotation applied)
    height: int | None = None
    # [[x, y] x4]. none_as_null: Python None is stored as SQL NULL, not JSON 'null'
    court_corners: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSONB(none_as_null=True))
    )
    # Which detector made the current rallies ("v1", "v1.1-2026-11", ...); set by
    # the worker with them. Null until the video has been processed.
    detector_version: str | None = Field(default=None, max_length=30)
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
    failure_reason: FailureReason | None = Field(
        description="Why it failed, for the user (only when status is 'failed')"
    )
    duration: float | None
    fps: float | None
    width: int | None
    height: int | None
    court_corners: list[Point] | None
    detector_version: str | None = Field(
        description="Detector that made the rallies, e.g. 'v1' or 'v1.1-2026-11' "
        "(null until processed). Re-analyzing uses the current detector."
    )
    created_at: datetime
    updated_at: datetime
    # Filled in by crud (one aggregate query, also for whole lists)
    rally_count: int = Field(description="Number of detected rallies")
    play_seconds: float = Field(
        description="Total rally time in seconds (sum of rally durations): "
        "the match without the breaks"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def court_out_of_frame(self) -> bool:
        """Warning, not an error: part of the court is well outside the video, so
        results may be less accurate. The app should tell the user and let them
        decide. Computed here so the web and mobile apps can't disagree."""
        return self.court_corners is not None and is_court_out_of_frame(
            self.court_corners
        )


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
    # 'rules' | 'model' (found by a detector) | 'user' (added by hand)
    source: str = Field(default="rules", max_length=10)
    # The detector's bounds, kept when the user corrects the rally; null for rallies
    # the user added. Corrections copy them, so detector output vs. the user's
    # version stays available as training labels.
    detected_start: float | None = None
    detected_end: float | None = None
    # The detector's probability (0-1) that this is a real rally; null for v1 (rules)
    # and for rallies the user added.
    confidence: float | None = None
    # Shown as a "maybe" the user confirms or deletes (detector was unsure). Cleared
    # when the user confirms, edits, splits or merges it.
    is_maybe: bool = Field(default=False, sa_column_kwargs={"server_default": "false"})
    created_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),
    )
    video: Video | None = Relationship(back_populates="rallies")


class RallyPublic(SQLModel):
    id: int
    number: int = Field(
        description="1, 2, 3... in time order; changes when rallies are added, "
        "split, merged or deleted before this one"
    )
    start: float
    end: float
    confidence: float | None = Field(
        default=None,
        description="Detector's probability (0-1) that this is a real rally; null "
        "for the v1 detector and for rallies the user added",
    )
    is_maybe: bool = Field(
        default=False,
        description="The detector was unsure: show it as a 'maybe' the user confirms "
        "(POST /rallies/{id}/confirm) or deletes. Cleared by any edit.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


class RalliesPublic(SQLModel):
    data: list[RallyPublic]
    count: int


# Rally corrections ----------------------------------------------------------

MIN_RALLY_SECONDS = 0.1  # shortest rally (or part of a split) a user can make

CorrectionAction = Literal["add", "edit", "delete", "split", "merge", "confirm"]


class Correction(SQLModel, table=True):
    """The log of the user's rally corrections, kept as future training data: the
    detector's bounds (original_*) next to the user's (edited_*). One row per rally
    involved; a split or a merge writes two rows with the same operation_id. No
    foreign keys to rally: the rows outlive the rallies they describe."""

    id: int | None = Field(default=None, primary_key=True)
    video_id: uuid.UUID = Field(
        foreign_key="video.id", nullable=False, ondelete="CASCADE", index=True
    )
    operation_id: uuid.UUID = Field(index=True)  # one per API call
    user_id: uuid.UUID | None = Field(
        default=None, foreign_key="user.id", ondelete="SET NULL"
    )
    action: str = Field(max_length=10)  # CorrectionAction
    rally_id: int | None = None  # the rally before the change (null for add)
    result_rally_id: int | None = None  # the rally after it (null for delete)
    # The detector's bounds for that rally (null: the user added it)
    original_start: float | None = None
    original_end: float | None = None
    # Bounds after the change (null for delete)
    edited_start: float | None = None
    edited_end: float | None = None
    created_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),
    )


class RallyTimes(SQLModel):
    """Add a rally, or set an existing one's start/end (seconds)."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def check_length(self) -> Self:
        if self.end - self.start < MIN_RALLY_SECONDS:
            raise ValueError(f"end must be at least {MIN_RALLY_SECONDS} s after start")
        return self


class RallySplit(SQLModel):
    at: float = Field(gt=0, description="Seconds; becomes the end of the first part "
                      "and the start of the second")  # fmt: skip


class RallyMerge(SQLModel):
    rally_ids: list[int] = Field(
        min_length=2, max_length=2, description="Two rallies next to each other"
    )


class ReelImpact(SQLModel):
    """How a reel was adjusted to a rally change."""

    id: uuid.UUID
    title: str
    updated_at: datetime = Field(description="New version: an open editor must reload")
    clips_removed: int = Field(description="Clips dropped (their rally is gone, or "
                               "nothing of them was left inside the new bounds)")  # fmt: skip
    pieces_trimmed: int = Field(description="Pieces cut back to the new bounds")
    pieces_removed: int = Field(description="Pieces entirely outside the new bounds")


class RallyChange(SQLModel):
    """What a correction changed, so the app can update its list and tell the user
    (e.g. "Rally 5 split into 5 and 6; 1 reel updated")."""

    added: list[RallyPublic]
    updated: list[RallyPublic] = Field(description="Rallies whose start/end changed")
    deleted: list[int] = Field(description="Ids of rallies that no longer exist")
    rallies: list[RallyPublic] = Field(
        description="All the video's rallies after the change, numbered in time order"
    )
    reels: list[ReelImpact] = Field(description="Reels that had to be adjusted")


# ---------------------------------------------------------------------------
# Errors the app handles specially carry a machine-readable code
# ---------------------------------------------------------------------------


class ErrorCode(StrEnum):
    TRAJECTORY_MISSING = "trajectory_missing"  # analyzed before trails existed
    REEL_CHANGED = "reel_changed"  # stale save: the reel was saved elsewhere
    REEL_MISSING = "reel_missing"  # the video has no reel yet
    THUMBNAILS_MISSING = "thumbnails_missing"  # analyzed before thumbnails existed


class ErrorWithCode(SQLModel):
    """{"detail": "...", "code": "..."}: detail is for people, code for the app."""

    detail: str
    code: ErrorCode


# ---------------------------------------------------------------------------
# Shuttle trajectory (a file per video in storage, see app/trajectory.py)
# ---------------------------------------------------------------------------


class TrajectoryPoint(SQLModel):
    frame: int = Field(description="Frame index; the frame is at frame / fps seconds")
    t: float = Field(description="Seconds in the original video (frame / fps)")
    visible: bool = Field(description="Shuttle seen in this frame")
    x: float | None = Field(description="Fraction of the frame width; null if not seen")
    y: float | None = Field(
        description="Fraction of the frame height; null if not seen"
    )


class TrajectoryPublic(SQLModel):
    fps: float
    start: float
    end: float
    points: list[TrajectoryPoint] = Field(
        description="Every frame between start and end, in order"
    )


# ---------------------------------------------------------------------------
# Reels: non-destructive highlight edits. The original video is never changed; a
# reel is an edit list (JSON) that the app plays back. All times are seconds in the
# original video.
# ---------------------------------------------------------------------------

REEL_TITLE_MAX_LENGTH = 100
CAPTION_MAX_LENGTH = 80
MAX_REEL_ITEMS = 500  # clips / restored gaps / captions per reel
# The app's player may report a slightly different duration than ffprobe did
TRIM_TOLERANCE = 0.1  # seconds
MAX_LEAD = 3.0  # lead_in / lead_out: how far pieces may go past a rally's bounds
DEFAULT_LEAD = 1.0
# Float slack when comparing a piece with its rally's bounds: the app computes
# "rally start - lead_in" in JavaScript, which may differ in the last digits
BOUND_TOLERANCE = 0.01


class EditModel(BaseModel):
    # Unknown fields are an error, not silently dropped: a typo in the app would
    # otherwise lose the user's edit without anyone noticing
    model_config = ConfigDict(extra="forbid")


class TimeRange(EditModel):
    start: float = PydanticField(ge=0)
    end: float = PydanticField(ge=0)

    @model_validator(mode="after")
    def check_order(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class Piece(TimeRange):
    slow_mo: bool = False


class ReelClip(EditModel):
    rally_id: int = PydanticField(description="The rally this clip shows")
    pieces: list[Piece] = PydanticField(
        default=[],
        max_length=MAX_REEL_ITEMS,
        description="The parts of the rally to show: sorted, not overlapping (they "
        "may touch, e.g. after a split), within the rally's bounds widened by the "
        "reel's lead_in/lead_out. A clip without pieces is dropped when saved.",
    )

    @model_validator(mode="after")
    def check_pieces(self) -> Self:
        for before, after in itertools.pairwise(self.pieces):
            if after.start < before.end:
                raise ValueError("pieces must be sorted and must not overlap")
        return self


class Caption(TimeRange):
    text: str = PydanticField(min_length=1, max_length=CAPTION_MAX_LENGTH)


class ReelOverlays(EditModel):
    trail: bool = False  # draw the shuttle trail (needs GET /videos/{id}/trajectory)
    captions: list[Caption] = PydanticField(default=[], max_length=MAX_REEL_ITEMS)


Lead = Annotated[float, PydanticField(ge=0, le=MAX_LEAD)]


class ReelEdit(EditModel):
    """The edit list, stored as JSON. `version` changes when the format does; an
    Alembic migration converts stored reels, so the code only knows the current one.
    v2 (this): clips are rallies cut into pieces. v1 had one trim per clip."""

    version: Literal[2] = 2
    view: Literal["cut", "full"] = "cut"
    format: Literal["16:9", "9:16"] = "16:9"
    lead_in: Lead = PydanticField(
        default=DEFAULT_LEAD,
        description="Seconds a piece may start before its rally's detected start "
        "(0-3; the detector can be 1-2 s off)",
    )
    lead_out: Lead = PydanticField(
        default=DEFAULT_LEAD,
        description="Seconds a piece may end after its rally's detected end (0-3)",
    )
    clips: list[ReelClip] = PydanticField(
        default=[], max_length=MAX_REEL_ITEMS, description="In reel order"
    )
    restored_gaps: list[TimeRange] = PydanticField(
        default=[],
        max_length=MAX_REEL_ITEMS,
        description="Breaks between rallies shown again in cut view",
    )
    overlays: ReelOverlays = PydanticField(default_factory=ReelOverlays)

    def without_empty_clips(self) -> ReelEdit:
        return self.model_copy(update={"clips": [c for c in self.clips if c.pieces]})

    def problems(
        self, *, duration: float | None, rallies: dict[int, tuple[float, float]]
    ) -> list[tuple[tuple[str | int, ...], str]]:
        """Checks that need the video: (location in the edit, message) for each
        rally_id that isn't one of the video's rallies, each piece outside its
        rally's padded bounds, and each time past the end of the video."""
        found: list[tuple[tuple[str | int, ...], str]] = []
        limit = None if duration is None else duration + TRIM_TOLERANCE
        past_end = f"must be within the video (0-{duration or 0:.2f} s)"
        for i, clip in enumerate(self.clips):
            bounds = rallies.get(clip.rally_id)
            if bounds is None:
                found.append((("clips", i, "rally_id"), "not a rally of this video"))
                continue
            lo, hi = bounds[0] - self.lead_in, bounds[1] + self.lead_out
            outside = (
                f"must be within the rally ({bounds[0]:.2f}-{bounds[1]:.2f} s) "
                f"widened by lead_in/lead_out ({max(lo, 0):.2f}-{hi:.2f} s)"
            )
            for j, piece in enumerate(clip.pieces):
                if piece.start < lo - BOUND_TOLERANCE:
                    found.append((("clips", i, "pieces", j, "start"), outside))
                if piece.end > hi + BOUND_TOLERANCE:
                    found.append((("clips", i, "pieces", j, "end"), outside))
                elif limit is not None and piece.end > limit:
                    found.append((("clips", i, "pieces", j, "end"), past_end))
        for i, gap in enumerate(self.restored_gaps):
            if limit is not None and gap.end > limit:
                found.append((("restored_gaps", i, "end"), past_end))
        for i, caption in enumerate(self.overlays.captions):
            if limit is not None and caption.end > limit:
                found.append((("overlays", "captions", i, "end"), past_end))
        return found


ReelTitle = Annotated[
    str, PydanticField(min_length=1, max_length=REEL_TITLE_MAX_LENGTH)
]


class Reel(SQLModel, table=True):
    """One reel per video for the beta: at most one non-deleted reel per video_id
    (a partial unique index; soft-deleted ones don't count)."""

    __table_args__ = (
        Index(
            "uq_reel_video_id_live",
            "video_id",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    video_id: uuid.UUID = Field(
        foreign_key="video.id", nullable=False, ondelete="CASCADE", index=True
    )
    # The video's owner (also when a superuser made the reel): a reel belongs to
    # whoever owns the footage. Stored, not joined, so "my reels" is one query.
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE", index=True
    )
    title: str = Field(max_length=REEL_TITLE_MAX_LENGTH)
    edit: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))  # ReelEdit
    created_at: datetime = Field(
        default_factory=get_datetime_utc, sa_type=DateTime(timezone=True)
    )
    # Set on every save; also the version token for stale-save checks (crud.update_reel)
    updated_at: datetime = Field(
        default_factory=get_datetime_utc, sa_type=DateTime(timezone=True)
    )
    # Soft delete: hidden from the API, kept in the database
    deleted_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))


class ReelCreate(SQLModel):
    title: ReelTitle
    edit: ReelEdit = Field(default_factory=ReelEdit)


class ReelSave(SQLModel):
    """PUT /videos/{id}/reel: create the video's reel, or replace its title and edit
    list."""

    title: ReelTitle
    edit: ReelEdit
    updated_at: AwareDatetime | None = Field(
        default=None,
        description="null: create the reel (the video has none yet). Otherwise the "
        "reel's updated_at as you loaded it. If the reel changed since (saved, "
        "created or deleted elsewhere), nothing is saved: 409, code reel_changed",
    )


class ReelUpdate(SQLModel):
    """Full replace of the title and edit list."""

    title: ReelTitle
    edit: ReelEdit
    updated_at: AwareDatetime = Field(
        description="The reel's updated_at as you loaded it. If it was saved "
        "elsewhere since then, the save is refused (409, code reel_changed)"
    )


class ReelPublic(SQLModel):
    id: uuid.UUID
    video_id: uuid.UUID
    owner_id: uuid.UUID
    title: str
    edit: ReelEdit
    created_at: datetime
    updated_at: datetime


class ReelsPublic(SQLModel):
    data: list[ReelPublic]
    count: int


class Health(SQLModel):
    status: Literal["ok", "error"]
    database: Literal["ok", "unreachable"]
    queued: int | None = Field(default=None, description="Videos waiting for a worker")
    processing: int | None = Field(default=None, description="Videos being processed")
    oldest_queued_seconds: float | None = Field(
        default=None,
        description="How long the oldest queued video has waited (since it was "
        "queued or re-queued); null when nothing is queued",
    )


class ThumbnailUrls(SQLModel):
    """Signed links (12 h) to a video's thumbnail strip. The index says where each
    second's tile is: tile i (time i * interval) is in sheet i // (columns * rows),
    at column i % columns and row (i // columns) % rows."""

    index_url: str
    sheet_urls: list[str] = Field(description="In the order of the index's `sheets`")
    expires_at: datetime


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
