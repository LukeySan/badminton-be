"""
Make thumbnail strips for videos analyzed before the worker made them.

    uv run --package worker python worker/backfill_thumbnails.py             # all missing
    uv run --package worker python worker/backfill_thumbnails.py --dry-run   # just list
    uv run --package worker python worker/backfill_thumbnails.py --video <id> --force

Only `done` videos (a video being processed gets its strip from the worker).
Uses the analysis copy (work/analysis.mp4, or work/video_720.mp4 from older runs)
when it's still there, else the original. Needs ffmpeg, not torch, and can run while workers are running. Safe to
run again: videos that already have a strip are skipped unless --force.
"""

import argparse
import logging
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from app import thumbnails
from app.core.config import settings
from app.core.db import engine as default_engine
from app.models import Video

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thumbnail_strip import generate_thumbnails  # noqa: E402

logger = logging.getLogger("backfill")


ANALYSIS_COPIES = ("analysis.mp4", "video_720.mp4")  # current name, then older runs'


def source_video(video: Video) -> Path:
    work = settings.STORAGE_DIR / str(video.id) / "work"
    for name in ANALYSIS_COPIES:
        if (work / name).exists():
            return work / name
    return settings.STORAGE_DIR / video.storage_path


def backfill(
    engine: Engine = default_engine,
    *,
    video_ids: Sequence[uuid.UUID] = (),
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, list[uuid.UUID]]:
    """Returns the video ids by outcome: made, skipped (already had a strip),
    failed (e.g. the file is missing)."""
    with Session(engine) as session:
        query = select(Video).where(col(Video.status) == "done")
        if video_ids:
            query = query.where(col(Video.id).in_(video_ids))
        videos = session.exec(query.order_by(col(Video.created_at))).all()
        for video in videos:
            session.expunge(video)
    outcome: dict[str, list[uuid.UUID]] = {"made": [], "skipped": [], "failed": []}
    for video in videos:
        short = str(video.id)[:8]
        if thumbnails.read_index(video.id) is not None and not force:
            outcome["skipped"].append(video.id)
            continue
        src = source_video(video)
        if dry_run:
            logger.info("[%s] would make thumbnails from %s", short, src.name)
            outcome["made"].append(video.id)
            continue
        started = time.monotonic()
        try:
            generate_thumbnails(str(src), str(thumbnails.thumbnails_dir(video.id)))
        except Exception as e:  # one bad video shouldn't stop the rest
            logger.warning("[%s] failed: %s: %s", short, type(e).__name__, e)
            outcome["failed"].append(video.id)
            continue
        logger.info(
            "[%s] thumbnails made from %s in %.0fs",
            short,
            src.name,
            time.monotonic() - started,
        )
        outcome["made"].append(video.id)
    return outcome


def main(engine: Engine = default_engine) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", type=uuid.UUID, action="append", default=[],
                        help="only this video (repeatable)")  # fmt: skip
    parser.add_argument("--force", action="store_true", help="remake existing strips")
    parser.add_argument("--dry-run", action="store_true", help="list, don't make")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    outcome = backfill(
        engine, video_ids=args.video, force=args.force, dry_run=args.dry_run
    )
    verb = "to make" if args.dry_run else "made"
    logger.info(
        "%d %s, %d already had thumbnails, %d failed",
        len(outcome["made"]),
        verb,
        len(outcome["skipped"]),
        len(outcome["failed"]),
    )
    if outcome["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
