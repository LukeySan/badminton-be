"""
Database engine/session, plus the status changes shared by the API and the worker.

Status changes are single conditional UPDATEs (update(...).where(status in ...)), not
"load the row, check it in Python, save it": that leaves a gap between check and write
where two clicks on "process", or two workers claiming a job, could both succeed.
"""

from sqlalchemy import create_engine, event, update
from sqlalchemy.orm import sessionmaker

from .config import DB_PATH
from .models import Base, Video, utcnow

# Where each status may move to. "Queued" is allowed again from done/failed so a
# video can be retried or reprocessed after fixing the court.
CAN_QUEUE_FROM = ("uploaded", "done", "failed")
CAN_EDIT_COURT_IN = ("uploaded", "done", "failed")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
# check_same_thread=False: FastAPI may open the session and run the handler on different threads.
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False, "timeout": 10})
SessionLocal = sessionmaker(engine)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(conn, _record):
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")  # API reads don't block the worker's writes


def init():
    # Fine for the beta; switch to Alembic migrations once there's data worth keeping.
    Base.metadata.create_all(engine)


def get_session():
    """FastAPI dependency: one session per request."""
    with SessionLocal() as session:
        yield session


def set_court(session, video_id, corners):
    """Returns False if the video is queued/processing (or doesn't exist)."""
    result = session.execute(
        update(Video)
        .where(Video.id == video_id, Video.status.in_(CAN_EDIT_COURT_IN))
        .values(court_corners=corners, updated_at=utcnow())
    )
    session.commit()
    return result.rowcount == 1


def queue(session, video_id):
    """Returns False if the video can't be queued right now (or doesn't exist)."""
    result = session.execute(
        update(Video)
        .where(Video.id == video_id, Video.status.in_(CAN_QUEUE_FROM), Video.court_corners.is_not(None))
        .values(status="queued", progress=0.0, error=None, updated_at=utcnow())
    )
    session.commit()
    return result.rowcount == 1
