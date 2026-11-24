"""
Worker loop: takes queued jobs from the database: reel exports (rendered with
ffmpeg, worker/reel_export.py) first, because they are short, then videos (rally
analysis). On macOS, put FriBiDi on the library path first (Thai text overlays need
it, see worker/caption_image.py; the worker won't start without it):

    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:/usr/local/lib:/usr/lib
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
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError
from sqlmodel import Session

from app import crud, thumbnails, trajectory
from app.core.config import settings
from app.core.db import engine as default_engine
from app.models import EXPORT_SIZES, ExportFailureReason, FailureReason, Video

WORKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKER_DIR))  # for `import pipeline`, `import job_errors`

import caption_image  # noqa: E402
import reel_export  # noqa: E402
from job_errors import (  # noqa: E402
    CourtNotVisible,
    PipelineError,
    SourceUnreadable,
    VideoUnreadable,
)
from thumbnail_strip import replace_dir  # noqa: E402
from trajectory_file import FILENAME as TRAJECTORY_FILENAME  # noqa: E402

HEARTBEAT_SECONDS = 5.0
STALE_AFTER = timedelta(minutes=2)
POLL_SECONDS = 5.0
TRACKING_LOG_STEP = 10  # log tracking progress every 10%

logger = logging.getLogger("worker")


class Detection(TypedDict):
    """What the pipeline returns (detector.detect_rallies): the detector's version
    and its rallies, [{start, end, duration, confidence, is_maybe}] in seconds."""

    detector_version: str
    rallies: list[dict[str, Any]]


# (video file, corner fractions, work dir, on_progress(0-1), log(line)) -> Detection
Processor = Callable[
    [Path, list[list[float]], Path, Callable[[float], None], Callable[[str], None]],
    Detection,
]

# Pipeline errors -> what the user sees. Any other exception is a bug: UNKNOWN.
FAILURE_REASONS: dict[type[PipelineError], FailureReason] = {
    VideoUnreadable: FailureReason.VIDEO_UNREADABLE,
    CourtNotVisible: FailureReason.COURT_NOT_VISIBLE,
}


def failure_reason(e: Exception) -> FailureReason:
    for error_type, reason in FAILURE_REASONS.items():
        if isinstance(e, error_type):
            return reason
    return FailureReason.UNKNOWN


def export_failure_reason(e: Exception) -> ExportFailureReason:
    if isinstance(e, SourceUnreadable):
        return ExportFailureReason.SOURCE_UNREADABLE
    return ExportFailureReason.UNKNOWN


class ExportJob(NamedTuple):
    """A claimed export: what run_export needs, read before the session closes."""

    id: uuid.UUID
    video_id: uuid.UUID
    claim_id: uuid.UUID
    title: str
    format: str
    quality: str
    snapshot: dict[str, Any]
    source: Path  # the original upload
    attempts: int


# (original, snapshot, width, height, output file, on_progress(0-1), log(line))
#   -> {duration, size_bytes}
ExportRenderer = Callable[
    [Path, dict[str, Any], int, int, Path, Callable[[float], None],
     Callable[[str], None]],
    reel_export.RenderResult,
]  # fmt: skip


def render_export(
    source: Path,
    snapshot: dict[str, Any],
    width: int,
    height: int,
    out_path: Path,
    on_progress: Callable[[float], None],
    log: Callable[[str], None],
) -> reel_export.RenderResult:
    return reel_export.render_export(
        str(source), snapshot, width, height, str(out_path), on_progress, log
    )


def export_output_path(video_id: uuid.UUID, export_id: uuid.UUID) -> str:
    """Relative to STORAGE_DIR: next to the video, so deleting it removes them."""
    return f"{video_id}/exports/{export_id}.mp4"


def fmt_duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


class JobLog:
    """One job's log lines, prefixed with the video's short id so they can be told
    apart when several workers log at once:  [3f2a1c9e] tracking 40%"""

    def __init__(self, video_id: uuid.UUID) -> None:
        self.prefix = f"[{str(video_id)[:8]}]"

    def info(self, message: str) -> None:
        logger.info("%s %s", self.prefix, message)

    def warning(self, message: str) -> None:
        logger.warning("%s %s", self.prefix, message)

    def exception(self, message: str) -> None:
        logger.exception("%s %s", self.prefix, message)  # with traceback


def publish_outputs(work_dir: Path, video_id: uuid.UUID) -> None:
    """Move the pipeline's files for the app (shuttle trajectory, thumbnail strip)
    from the work folder to where the API reads them. Renames, so the API sees the
    old version or the new one, never half of one. Before the video is marked done,
    so "done" always comes with the new files; a failed run keeps the old ones."""
    src = work_dir / TRAJECTORY_FILENAME
    if src.exists():
        os.replace(src, trajectory.trajectory_path(video_id))
    src = work_dir / thumbnails.DIRNAME
    if src.is_dir():
        replace_dir(str(src), str(thumbnails.thumbnails_dir(video_id)))


class JobLost(Exception):
    """The job was taken away from this worker (re-queued as stale, or deleted)."""


def run_pipeline(
    video_path: Path,
    corners: list[list[float]],
    work_dir: Path,
    on_progress: Callable[[float], None],
    log: Callable[[str], None],
    batch_size: int = 4,
    detector: str = settings.DETECTOR_VERSION,
) -> Detection:
    # Imported here so torch/opencv only load when there's actually a job,
    # and so tests can run the loop with a fake processor without them.
    from pipeline import process_video

    result: Detection = process_video(
        str(video_path),
        corners,
        str(work_dir),
        batch_size=batch_size,
        on_progress=on_progress,
        log=log,
        detector=detector,
    )
    return result


def parse_detection(result: object) -> tuple[str, list[crud.DetectedRally]]:
    """The pipeline's result -> (detector version, rallies), with a clear error if
    it isn't a Detection.

    Why the check: the worker imports `pipeline` once, but the code on disk can
    change under a running worker (branch switch, pull). That happened: an old loop
    (expecting a list of rallies) got v1.1's {detector_version, rallies} dict and
    crashed with "string indices must be integers". Restart the worker after
    changing the code; this turns the mix-up into a failed job, not a dead worker."""
    if not (
        isinstance(result, dict)
        and isinstance(result.get("detector_version"), str)
        and isinstance(result.get("rallies"), list)
    ):
        raise TypeError(
            f"pipeline returned {type(result).__name__}, expected "
            "{detector_version, rallies}: restart the worker after code changes"
        )
    rallies = [
        crud.DetectedRally(
            float(r["start"]),
            float(r["end"]),
            None if r.get("confidence") is None else float(r["confidence"]),
            bool(r.get("is_maybe", False)),
        )
        for r in result["rallies"]
    ]
    return result["detector_version"], rallies


def record_failure(
    engine: Engine,
    error: Exception,
    log: JobLog,
    reason: StrEnum,
    save: Callable[[Session, str], bool],
) -> None:
    """Mark the job failed: a short reason for the user (`reason`), the details
    internally; save(session, details) writes both and returns False if the job is
    no longer ours. Never raises: if even that can't be saved (database down), the
    job stays 'processing' without a heartbeat, so the stale re-queue picks it up
    again after STALE_AFTER (and gives up after MAX_ATTEMPTS)."""
    details = f"{type(error).__name__}: {error}"
    if isinstance(error, PipelineError):
        log.warning(f"failed: {reason} ({details})")
    else:  # unexpected: a bug, so keep the traceback
        log.exception(f"failed: {reason} ({details})")
    try:
        with Session(engine) as session:
            ours = save(session, details)
    except Exception:
        log.exception(
            "could not save the failure; the job goes back to the queue once its "
            "heartbeat is stale"
        )
        return
    if not ours:
        log.warning("lost: another worker has this job now; failure not saved")


class Heartbeat(threading.Thread):
    """Writes progress + heartbeat every `interval` seconds until stopped.

    A thread, not just the progress callback: the downscale step and model loading
    report no progress and can take minutes on a long video.
    write(session, progress) does the update and returns False once the job is no
    longer ours. `label` names the slow step in the log ("tracking 40%"), `scale`
    keeps the last part of the progress bar for saving the results.
    """

    def __init__(
        self,
        engine: Engine,
        write: Callable[[Session, float], bool],
        interval: float,
        log: JobLog,
        label: str = "tracking",
        scale: float = 0.95,
    ) -> None:
        super().__init__(daemon=True, name=f"heartbeat{log.prefix}")
        self.engine, self.write = engine, write
        self.interval = interval
        self.log = log
        self.label, self.scale = label, scale
        self.progress = 0.0
        self.logged_percent = 0
        self.lost = threading.Event()
        self._stop_event = threading.Event()

    def on_progress(self, fraction: float) -> None:
        """Pipeline callback (progress 0-1 of the slow step). Raises JobLost to stop
        the pipeline early."""
        if self.lost.is_set():
            raise JobLost
        self.progress = self.scale * fraction
        percent = int(fraction * 100) // TRACKING_LOG_STEP * TRACKING_LOG_STEP
        if percent > self.logged_percent:
            self.logged_percent = percent
            self.log.info(f"{self.label} {percent}%")

    def beat(self) -> None:
        try:
            with Session(self.engine) as session:
                ours = self.write(session, self.progress)
        except OperationalError:
            # Database briefly unreachable: keep working, try again next beat
            self.log.warning("heartbeat failed (database unreachable); retrying")
            return
        if not ours:
            self.lost.set()

    def run(self) -> None:
        while not self._stop_event.wait(self.interval) and not self.lost.is_set():
            self.beat()

    def stop(self) -> None:
        self._stop_event.set()
        self.join()


def rescue_stale_jobs(session: Session, stale_after: timedelta) -> None:
    for kind, rescue in (
        ("exports", crud.requeue_stale_exports),
        ("videos", crud.requeue_stale_videos),
    ):
        requeued, gave_up = rescue(session=session, stale_after=stale_after)
        if requeued or gave_up:
            logger.warning(
                "stale %s (worker stopped responding): %d re-queued, %d failed",
                kind,
                requeued,
                gave_up,
            )


def claim_export(session: Session, worker_name: str) -> ExportJob | None:
    export = crud.claim_next_export(session=session, claimed_by=worker_name)
    if export is None:
        return None
    assert export.claim_id is not None
    video = session.get(Video, export.video_id)
    assert video is not None  # exports go with their video (database cascade)
    return ExportJob(
        id=export.id,
        video_id=export.video_id,
        claim_id=export.claim_id,
        title=export.title,
        format=export.format,
        quality=export.quality,
        snapshot=export.snapshot,
        source=settings.STORAGE_DIR / video.storage_path,
        attempts=export.attempts,
    )


def process_one(
    processor: Processor,
    worker_name: str,
    engine: Engine = default_engine,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    stale_after: timedelta = STALE_AFTER,
    renderer: ExportRenderer = render_export,
) -> bool:
    """Rescue stale jobs, then claim and process one job: an export if one is
    queued (they take seconds to minutes), else a video.
    Returns False if there was nothing to do.

    Logs one line per phase: claimed, prepared, tracking N%, detecting rallies,
    then done / failed (with the reason) / lost / stopped."""
    with Session(engine) as session:
        rescue_stale_jobs(session, stale_after)
        export = claim_export(session, worker_name)
    if export is not None:
        return run_export(export, renderer, engine, heartbeat_seconds)

    with Session(engine) as session:
        video = crud.claim_next_video(session=session, claimed_by=worker_name)
        if video is None:
            return False
        video_id, claim_id = video.id, video.claim_id
        assert claim_id is not None
        video_path = settings.STORAGE_DIR / video.storage_path
        corners = video.court_corners
        log = JobLog(video_id)
        log.info(
            f"claimed: {video.filename}, {(video.duration or 0) / 60:.1f} min, "
            f"attempt {video.attempts}/{crud.MAX_ATTEMPTS}"
        )

    work_dir = settings.STORAGE_DIR / str(video_id) / "work"
    heartbeat = Heartbeat(
        engine,
        lambda session, progress: crud.heartbeat_video(
            session=session, video_id=video_id, claim_id=claim_id, progress=progress
        ),
        heartbeat_seconds,
        log,
    )
    heartbeat.start()
    started = time.monotonic()
    # Everything after the claim is inside this try, not just the pipeline call:
    # an error while reading the result or saving it must fail THIS job, not kill
    # the worker (it did once: see parse_detection).
    try:
        if corners is None:
            raise ValueError("The video has no court corners")
        result = processor(
            video_path, corners, work_dir, heartbeat.on_progress, log.info
        )
        heartbeat.stop()
        # Checked before publishing, so a bad result keeps the old files
        version, rallies = parse_detection(result)
        publish_outputs(work_dir, video_id)
        with Session(engine) as session:
            saved = crud.finish_video(
                session=session,
                video_id=video_id,
                claim_id=claim_id,
                rallies=rallies,
                detector_version=version,
                source="rules" if version == "v1" else "model",
            )
    except JobLost:
        heartbeat.stop()
        log.warning("lost: another worker has this job now; stopped")
        return True
    except KeyboardInterrupt:
        heartbeat.stop()
        with Session(engine) as session:
            crud.release_video(session=session, video_id=video_id, claim_id=claim_id)
        log.info("stopped (Ctrl+C); back in the queue")
        raise
    except Exception as e:
        heartbeat.stop()
        reason = failure_reason(e)
        record_failure(
            engine,
            e,
            log,
            reason,
            lambda session, details: crud.fail_video(
                session=session,
                video_id=video_id,
                claim_id=claim_id,
                reason=reason,
                error=details,
            ),
        )
        return True

    if saved:
        play = sum(r.end - r.start for r in rallies)
        maybes = sum(1 for r in rallies if r.is_maybe)
        log.info(
            f"done: {len(rallies)} rallies"
            + (f", {maybes} maybe" if maybes else "")
            + f" ({play / 60:.1f} min of play) "
            f"in {fmt_duration(time.monotonic() - started)}, detector {version}"
        )
    else:
        log.warning("lost: results discarded, another worker has this job now")
    return True


def run_export(
    job: ExportJob, renderer: ExportRenderer, engine: Engine, heartbeat_seconds: float
) -> bool:
    """Render a claimed export and save the result. Like a video job: a failure
    fails this export only; deleting the export (or its video) meanwhile stops the
    render at the next progress update and discards the file."""
    log = JobLog(job.id)
    expected = float(job.snapshot.get("duration", 0))
    log.info(
        f"claimed export: {job.title!r}, {job.format} {job.quality}, "
        f"{expected:.1f} s, attempt {job.attempts}/{crud.MAX_ATTEMPTS}"
    )
    width, height = EXPORT_SIZES[(job.format, job.quality)]
    relative = export_output_path(job.video_id, job.id)
    out_path = settings.STORAGE_DIR / relative
    heartbeat = Heartbeat(
        engine,
        lambda session, progress: crud.heartbeat_export(
            session=session, export_id=job.id, claim_id=job.claim_id, progress=progress
        ),
        heartbeat_seconds,
        log,
        label="rendering",
        scale=0.99,
    )
    heartbeat.start()
    started = time.monotonic()
    try:
        result = renderer(
            job.source, job.snapshot, width, height, out_path,
            heartbeat.on_progress, log.info,
        )  # fmt: skip
        heartbeat.stop()
        with Session(engine) as session:
            saved = crud.finish_export(
                session=session,
                export_id=job.id,
                claim_id=job.claim_id,
                output_path=relative,
                size_bytes=int(result["size_bytes"]),
                duration=float(result["duration"]),
            )
    except JobLost:
        heartbeat.stop()
        out_path.unlink(missing_ok=True)
        log.warning("lost: deleted, or another worker has this export now; stopped")
        return True
    except KeyboardInterrupt:
        heartbeat.stop()
        with Session(engine) as session:
            crud.release_export(
                session=session, export_id=job.id, claim_id=job.claim_id
            )
        log.info("stopped (Ctrl+C); back in the queue")
        raise
    except Exception as e:
        heartbeat.stop()
        out_path.unlink(missing_ok=True)
        reason = export_failure_reason(e)
        record_failure(
            engine,
            e,
            log,
            reason,
            lambda session, details: crud.fail_export(
                session=session,
                export_id=job.id,
                claim_id=job.claim_id,
                reason=reason,
                error=details,
            ),
        )
        return True

    if saved:
        log.info(
            f"done: export {result['size_bytes'] / 1e6:.1f} MB, "
            f"{result['duration']:.1f} s, in {fmt_duration(time.monotonic() - started)}"
        )
    else:
        out_path.unlink(missing_ok=True)
        log.warning("lost: export discarded (deleted, or another worker has it now)")
    return True


TEXT_LAYOUT_HELP = (
    "Pillow has no complex text layout (raqm), which Thai text overlays need. Install "
    "FriBiDi: on macOS `brew install fribidi` and start the worker with "
    "DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:/usr/local/lib:/usr/lib; on "
    "Linux `apt install libfribidi0`."
)


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
    # Exports with Thai text would render wrongly without it: refuse to start
    if not caption_image.check_text_layout():
        logger.error("Not starting: %s", TEXT_LAYOUT_HELP)
        sys.exit(1)
    detector = settings.DETECTOR_VERSION
    # Load the detector's model now: a missing or mismatched model file should stop
    # the worker at start, not fail every video it claims.
    from detector import DETECTORS, preload

    preload(detector)
    # Import the whole pipeline now too, not on the first job: otherwise a worker
    # started before a branch switch runs its old loop with the new pipeline.
    import pipeline  # noqa: F401

    def processor(
        video_path: Path,
        corners: list[list[float]],
        work_dir: Path,
        on_progress: Callable[[float], None],
        log: Callable[[str], None],
    ) -> Detection:
        return run_pipeline(
            video_path,
            corners,
            work_dir,
            on_progress,
            log,
            batch_size=args.batch_size,
            detector=detector,
        )

    logger.info(
        "Worker %s started (detector %s), waiting for jobs",
        worker_name,
        DETECTORS[detector],
    )
    serve_forever(
        processor,
        worker_name,
        once=args.once,
        poll_seconds=args.poll,
        stale_after=timedelta(seconds=args.stale_after),
    )


def serve_forever(
    processor: Processor,
    worker_name: str,
    *,
    once: bool = False,
    poll_seconds: float = POLL_SECONDS,
    stale_after: timedelta = STALE_AFTER,
    engine: Engine = default_engine,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    renderer: ExportRenderer = render_export,
    wait: Callable[[float], None] = time.sleep,
) -> None:
    """Process jobs until Ctrl+C (or one job with once=True). Nothing a job does
    stops the loop; only Ctrl+C does. wait(seconds): how it waits for new jobs
    (tests pass their own, rather than patching time.sleep, which subprocess uses
    while ffmpeg/ffprobe run)."""
    try:
        while True:
            try:
                did_work = process_one(
                    processor,
                    worker_name,
                    engine=engine,
                    heartbeat_seconds=heartbeat_seconds,
                    stale_after=stale_after,
                    renderer=renderer,
                )
            except OperationalError as e:
                logger.error("Database unavailable (%s); retrying", e.orig)
                did_work = False
            except Exception:
                # Last line of defence: process_one handles job errors itself, so
                # this is a bug in the loop. Log it and keep serving the queue; a
                # job it left 'processing' is re-queued once its heartbeat is stale.
                logger.exception("Unexpected error in the worker loop; continuing")
                did_work = False
            if once and did_work:
                break
            if not did_work:
                wait(poll_seconds)
    except KeyboardInterrupt:
        logger.info("Worker stopped")


if __name__ == "__main__":
    main()
