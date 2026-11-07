"""
Worker loop: takes queued videos from the database and finds their rallies.

    uv run --package worker python worker/run_worker.py               # run until Ctrl+C
    uv run --package worker python worker/run_worker.py --once        # one job, then exit
    uv run --package worker python worker/run_worker.py --batch-size 16   # NVIDIA GPU

Several workers can run at once, on one machine or several pointing at the same
database: each job is claimed by exactly one of them (see crud.claim_next_video).

While a job runs, a background thread refreshes the video's heartbeat (updated_at)
and progress every few seconds. Any worker re-queues jobs whose heartbeat is older
than --stale-after, so a crashed worker's job gets picked up again.
"""

import argparse
import logging
import os
import socket
import sys
import threading
import time
import uuid
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError
from sqlmodel import Session

from app import crud
from app.core.config import settings
from app.core.db import engine as default_engine

WORKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKER_DIR))  # for `import pipeline`

HEARTBEAT_SECONDS = 5.0
STALE_AFTER = timedelta(minutes=2)
POLL_SECONDS = 5.0

logger = logging.getLogger("worker")

# (video file, corner fractions, work dir, on_progress(0-1)) -> [{start, end, ...}]
Processor = Callable[
    [Path, list[list[float]], Path, Callable[[float], None]], list[dict[str, Any]]
]


class JobLost(Exception):
    """The job was taken away from this worker (re-queued as stale, or deleted)."""


def run_pipeline(
    video_path: Path,
    corners: list[list[float]],
    work_dir: Path,
    on_progress: Callable[[float], None],
    batch_size: int = 4,
) -> list[dict[str, Any]]:
    # Imported here so torch/opencv only load when there's actually a job,
    # and so tests can run the loop with a fake processor without them.
    from pipeline import process_video

    rallies: list[dict[str, Any]] = process_video(
        str(video_path),
        corners,
        str(work_dir),
        batch_size=batch_size,
        on_progress=on_progress,
        log=logger.info,
    )
    return rallies


class Heartbeat(threading.Thread):
    """Writes progress + heartbeat every `interval` seconds until stopped.

    A thread, not just the progress callback: the downscale step and model loading
    report no progress and can take minutes on a long video.
    """

    def __init__(
        self, engine: Engine, video_id: uuid.UUID, claim_id: uuid.UUID, interval: float
    ) -> None:
        super().__init__(daemon=True, name=f"heartbeat-{video_id}")
        self.engine, self.video_id, self.claim_id = engine, video_id, claim_id
        self.interval = interval
        self.progress = 0.0
        self.lost = threading.Event()
        self._stop_event = threading.Event()

    def on_progress(self, fraction: float) -> None:
        """Pipeline callback. Raises JobLost to stop the pipeline early."""
        if self.lost.is_set():
            raise JobLost
        # Tracking is ~all of the work; keep 100% for when the rallies are saved
        self.progress = 0.95 * fraction

    def beat(self) -> None:
        try:
            with Session(self.engine) as session:
                ours = crud.heartbeat_video(
                    session=session,
                    video_id=self.video_id,
                    claim_id=self.claim_id,
                    progress=self.progress,
                )
        except OperationalError:
            # Database briefly unreachable: keep working, try again next beat
            logger.warning("Heartbeat failed for %s; retrying", self.video_id)
            return
        if not ours:
            logger.warning("Lost job %s (re-queued or deleted)", self.video_id)
            self.lost.set()

    def run(self) -> None:
        while not self._stop_event.wait(self.interval) and not self.lost.is_set():
            self.beat()

    def stop(self) -> None:
        self._stop_event.set()
        self.join()


def process_one(
    processor: Processor,
    worker_name: str,
    engine: Engine = default_engine,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    stale_after: timedelta = STALE_AFTER,
) -> bool:
    """Rescue stale jobs, then claim and process one video.
    Returns False if there was nothing to do."""
    with Session(engine) as session:
        requeued, failed = crud.requeue_stale_videos(
            session=session, stale_after=stale_after
        )
        if requeued or failed:
            logger.warning("Stale jobs: %d re-queued, %d failed", requeued, failed)
        video = crud.claim_next_video(session=session, claimed_by=worker_name)
        if video is None:
            return False
        video_id, claim_id = video.id, video.claim_id
        assert claim_id is not None
        video_path = settings.STORAGE_DIR / video.storage_path
        corners = video.court_corners
        logger.info(
            "Processing %s (%s, attempt %d)", video_id, video.filename, video.attempts
        )

    work_dir = settings.STORAGE_DIR / str(video_id) / "work"
    heartbeat = Heartbeat(engine, video_id, claim_id, heartbeat_seconds)
    heartbeat.start()
    started = time.monotonic()
    try:
        if corners is None:
            raise ValueError("The video has no court corners")
        rallies = processor(video_path, corners, work_dir, heartbeat.on_progress)
    except JobLost:
        heartbeat.stop()
        logger.warning("Stopped %s: another worker has it now", video_id)
        return True
    except KeyboardInterrupt:
        heartbeat.stop()
        with Session(engine) as session:
            crud.release_video(session=session, video_id=video_id, claim_id=claim_id)
        logger.info("Stopped; %s is back in the queue", video_id)
        raise
    except Exception as e:
        heartbeat.stop()
        logger.exception("Failed %s", video_id)
        with Session(engine) as session:
            crud.fail_video(
                session=session,
                video_id=video_id,
                claim_id=claim_id,
                error=f"{type(e).__name__}: {e}",
            )
        return True
    heartbeat.stop()

    with Session(engine) as session:
        saved = crud.finish_video(
            session=session,
            video_id=video_id,
            claim_id=claim_id,
            rallies=[(float(r["start"]), float(r["end"])) for r in rallies],
        )
    if saved:
        logger.info(
            "Done %s: %d rallies in %.0fs",
            video_id,
            len(rallies),
            time.monotonic() - started,
        )
    else:
        logger.warning("Discarded results for %s: another worker has it now", video_id)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--once", action="store_true", help="process one job and exit")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="4 on the Mac, 16 on the RTX 2070"
    )
    parser.add_argument(
        "--poll", type=float, default=POLL_SECONDS, help="seconds between queue checks"
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=STALE_AFTER.total_seconds(),
        help="seconds without a heartbeat before a job is re-queued",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    worker_name = f"{socket.gethostname()}:{os.getpid()}"

    def processor(
        video_path: Path,
        corners: list[list[float]],
        work_dir: Path,
        on_progress: Callable[[float], None],
    ) -> list[dict[str, Any]]:
        return run_pipeline(
            video_path, corners, work_dir, on_progress, batch_size=args.batch_size
        )

    logger.info("Worker %s started, waiting for jobs", worker_name)
    try:
        while True:
            try:
                did_work = process_one(
                    processor,
                    worker_name,
                    stale_after=timedelta(seconds=args.stale_after),
                )
            except OperationalError as e:
                logger.error("Database unavailable (%s); retrying", e.orig)
                did_work = False
            if args.once and did_work:
                break
            if not did_work:
                time.sleep(args.poll)
    except KeyboardInterrupt:
        logger.info("Worker stopped")


if __name__ == "__main__":
    main()
