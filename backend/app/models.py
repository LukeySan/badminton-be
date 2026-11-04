"""Database tables (SQLAlchemy 2.0 ORM). Shared by the API and, from milestone 3, the worker."""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow():
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """SQLite drops time zones, so values come back "naive" and the frontend would read
    them as local time. Always hand back UTC-aware datetimes."""
    impl = DateTime
    cache_ok = True

    def process_result_value(self, value, dialect):
        return value.replace(tzinfo=timezone.utc) if value and value.tzinfo is None else value


class Base(DeclarativeBase):
    pass


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # uuid4, set by the upload route
    filename: Mapped[str] = mapped_column(String)          # the user's original name, display only
    storage_path: Mapped[str] = mapped_column(String)      # relative to STORAGE_DIR
    status: Mapped[str] = mapped_column(String, default="uploaded")
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0-1 while processing
    error: Mapped[str | None] = mapped_column(Text)
    duration: Mapped[float | None] = mapped_column(Float)
    fps: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)     # as displayed (rotation applied)
    height: Mapped[int | None] = mapped_column(Integer)
    # [[x, y] x4] fractions 0-1. none_as_null: store Python None as SQL NULL, not the JSON text 'null'.
    court_corners: Mapped[list | None] = mapped_column(JSON(none_as_null=True))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)  # worker heartbeat

    rallies: Mapped[list["Rally"]] = relationship(
        back_populates="video", cascade="all, delete-orphan", order_by="Rally.start")


class Rally(Base):
    __tablename__ = "rallies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String, default="rules")  # 'rules' | 'model'
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    video: Mapped[Video] = relationship(back_populates="rallies")

    @property
    def duration(self):
        return round(self.end - self.start, 2)


class Correction(Base):
    """Not used yet: user edits to rallies, kept with the detector's original as training data."""
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rally_id: Mapped[int | None] = mapped_column(Integer)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    original_start: Mapped[float | None] = mapped_column(Float)
    original_end: Mapped[float | None] = mapped_column(Float)
    edited_start: Mapped[float | None] = mapped_column(Float)
    edited_end: Mapped[float | None] = mapped_column(Float)
    action: Mapped[str] = mapped_column(String)  # 'edit' | 'delete' | 'add'
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
