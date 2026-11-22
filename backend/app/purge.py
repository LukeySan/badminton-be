"""
Delete expired reel exports: soft-deletes exports requested more than
EXPORT_RETENTION_DAYS (7) ago and removes the files of every deleted export.
Safe to run any time and as often as you like; schedule it (cron, launchd) e.g.
hourly. From backend/:

    uv run python -m app.purge
"""

import logging
from datetime import timedelta

from sqlmodel import Session

from app import crud
from app.core.config import settings
from app.core.db import engine

logger = logging.getLogger("purge")


def purge(session: Session, retention: timedelta | None = None) -> tuple[int, int]:
    """Returns (exports deleted now, files removed)."""
    if retention is None:
        retention = timedelta(days=settings.EXPORT_RETENTION_DAYS)
    expired, files = crud.purge_exports(session=session, older_than=retention)
    removed = 0
    for relative in files:  # after the commit (see crud.purge_exports)
        path = settings.STORAGE_DIR / relative
        if path.is_file():
            path.unlink()
            removed += 1
    return expired, removed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with Session(engine) as session:
        expired, removed = purge(session)
    logger.info("purge: %d exports expired, %d files removed", expired, removed)


if __name__ == "__main__":
    main()
