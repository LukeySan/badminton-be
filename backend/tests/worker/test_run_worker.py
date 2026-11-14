"""The worker loop (worker/run_worker.py), with a fake processor instead of
TrackNet, so these tests need neither torch nor a real video."""

import sys
import time
import uuid
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, col, select, update

from app import crud, trajectory
from app.models import FailureReason, Rally, Video
from tests.crud.test_video_jobs import (  # noqa: F401  (empty_videos is a fixture)
    empty_videos,
    fetch,
    make_video,
    rallies_of,
    set_age,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "worker"))
import run_worker  # noqa: E402
from job_errors import CourtNotVisible, VideoUnreadable  # noqa: E402

ProgressFn = Callable[[float], None]
LogFn = Callable[[str], None]
FAST_BEAT = 0.05  # heartbeat interval for tests, seconds


def detection(
    *rallies: dict[str, Any], version: str = "v1.1-2026-11"
) -> run_worker.Detection:
    return {"detector_version": version, "rallies": list(rallies)}


def run(engine: Engine, processor: run_worker.Processor) -> bool:
    return run_worker.process_one(
        processor, "test-worker", engine=engine, heartbeat_seconds=FAST_BEAT
    )


def test_processes_a_queued_video(db: Session, engine: Engine) -> None:
    video_id = make_video(db)
    calls: list[tuple[Path, list[list[float]], Path]] = []

    def processor(
        video_path: Path,
        corners: list[list[float]],
        work_dir: Path,
        on_progress: ProgressFn,
        _log: LogFn,
    ) -> run_worker.Detection:
        calls.append((video_path, corners, work_dir))
        on_progress(1.0)
        return detection(
            {"start": 12.0, "end": 20.5, "duration": 8.5},
            {"start": 1.0, "end": 4.0},
        )

    assert run(engine, processor) is True

    video_path, corners, work_dir = calls[0]
    assert video_path.name == "original.mp4"
    assert work_dir.parent.name == str(video_id)
    assert corners == fetch(engine, video_id).court_corners
    done = fetch(engine, video_id)
    assert (done.status, done.progress, done.claimed_by) == ("done", 1.0, "test-worker")
    assert rallies_of(engine, video_id) == [(1.0, 4.0), (12.0, 20.5)]


def test_nothing_to_do(engine: Engine) -> None:
    def processor(*_: Any) -> run_worker.Detection:
        raise AssertionError("must not be called")

    assert run(engine, processor) is False


def test_heartbeat_writes_progress_while_processing(
    db: Session, engine: Engine
) -> None:
    video_id = make_video(db)
    seen: list[Video] = []

    def processor(
        _path: Path,
        _corners: list[list[float]],
        _dir: Path,
        on_progress: ProgressFn,
        _log: LogFn,
    ) -> run_worker.Detection:
        set_age(db, video_id, timedelta(minutes=1))
        seen.append(fetch(engine, video_id))
        on_progress(0.5)
        time.sleep(FAST_BEAT * 6)  # several heartbeats
        seen.append(fetch(engine, video_id))
        return detection()

    run(engine, processor)
    before, during = seen
    assert during.status == "processing"
    assert during.progress == pytest.approx(0.95 * 0.5)
    # the heartbeat moved updated_at forward from "1 minute ago"
    assert during.updated_at - before.updated_at > timedelta(seconds=50)


def test_failure_marks_video_failed_with_error(db: Session, engine: Engine) -> None:
    video_id = make_video(db)

    def processor(*_: Any) -> run_worker.Detection:
        raise RuntimeError("TrackNet checkpoint not found")

    assert run(engine, processor) is True
    failed = fetch(engine, video_id)
    assert failed.status == "failed"
    assert failed.error == "RuntimeError: TrackNet checkpoint not found"
    assert failed.failure_reason == FailureReason.UNKNOWN  # unexpected: generic text
    assert rallies_of(engine, video_id) == []


def test_failure_keeps_rallies_from_an_earlier_run(db: Session, engine: Engine) -> None:
    """Reprocessing that fails shouldn't wipe the rallies the user already had."""
    video_id = make_video(db)
    run(engine, lambda *_: detection({"start": 1.0, "end": 5.0}))
    with Session(engine) as session:
        assert crud.queue_video(session=session, video_id=video_id) == "queued"

    def processor(*_: Any) -> run_worker.Detection:
        raise ValueError("bad video")

    run(engine, processor)
    assert fetch(engine, video_id).status == "failed"
    assert rallies_of(engine, video_id) == [(1.0, 5.0)]


def test_video_without_court_fails(db: Session, engine: Engine) -> None:
    video_id = make_video(db)
    db.exec(update(Video).where(col(Video.id) == video_id).values(court_corners=None))
    db.commit()

    assert run(engine, lambda *_: detection()) is True
    failed = fetch(engine, video_id)
    assert failed.status == "failed"
    assert failed.error and "no court corners" in failed.error


def test_ctrl_c_puts_the_job_back(db: Session, engine: Engine) -> None:
    video_id = make_video(db)

    def processor(*_: Any) -> run_worker.Detection:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(engine, processor)
    back = fetch(engine, video_id)
    assert (back.status, back.attempts, back.claim_id) == ("queued", 0, None)


def test_stops_when_another_worker_takes_the_job(db: Session, engine: Engine) -> None:
    """If the job is taken over mid-run, the heartbeat notices and the next
    progress callback stops the pipeline; nothing gets written."""
    video_id = make_video(db)
    other_claim = uuid.uuid4()
    reached_end = False

    def processor(
        _path: Path,
        _corners: list[list[float]],
        _dir: Path,
        on_progress: ProgressFn,
        _log: LogFn,
    ) -> run_worker.Detection:
        nonlocal reached_end
        with Session(engine) as session:  # another worker took over
            session.exec(
                update(Video)
                .where(col(Video.id) == video_id)
                .values(claim_id=other_claim)
            )
            session.commit()
        time.sleep(FAST_BEAT * 6)
        on_progress(0.5)  # raises JobLost
        reached_end = True
        return detection({"start": 1.0, "end": 2.0})

    assert run(engine, processor) is True
    assert not reached_end
    now = fetch(engine, video_id)
    assert (now.status, now.claim_id) == ("processing", other_claim)
    assert rallies_of(engine, video_id) == []


def test_results_discarded_if_job_lost_at_the_end(db: Session, engine: Engine) -> None:
    """Taken over after the last progress callback: finish_video refuses."""
    video_id = make_video(db)
    other_claim = uuid.uuid4()

    def processor(*_: Any) -> run_worker.Detection:
        with Session(engine) as session:
            session.exec(
                update(Video)
                .where(col(Video.id) == video_id)
                .values(claim_id=other_claim)
            )
            session.commit()
        return detection({"start": 1.0, "end": 2.0})

    assert run(engine, processor) is True
    assert fetch(engine, video_id).claim_id == other_claim
    assert rallies_of(engine, video_id) == []


def test_rescues_a_crashed_workers_job(db: Session, engine: Engine) -> None:
    """A video left in processing by a dead worker is re-queued and finished."""
    video_id = make_video(
        db, status="processing", age=timedelta(minutes=10), attempts=1
    )
    assert run(engine, lambda *_: detection({"start": 3.0, "end": 9.0})) is True
    done = fetch(engine, video_id)
    assert (done.status, done.attempts) == ("done", 2)
    assert rallies_of(engine, video_id) == [(3.0, 9.0)]


# --- failure reasons and log lines -----------------------------------------------


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (
            VideoUnreadable("OpenCV can't open or decode original.mp4"),
            "Couldn't read this video",
        ),
        (
            CourtNotVisible("shuttle inside the court zone in 0.4% of frames"),
            "Court not visible",
        ),
        (ZeroDivisionError("float division by zero"), FailureReason.UNKNOWN.value),
    ],
)
def test_failure_reason_for_the_user_and_details_kept_internal(
    db: Session, engine: Engine, error: Exception, reason: str
) -> None:
    video_id = make_video(db)

    def processor(*_: Any) -> run_worker.Detection:
        raise error

    run(engine, processor)
    failed = fetch(engine, video_id)
    assert failed.status == "failed"
    assert failed.failure_reason == reason
    assert failed.error == f"{type(error).__name__}: {error}"


def worker_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "worker"]


def test_one_log_line_per_phase(
    db: Session, engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    video_id = make_video(db)
    db.exec(update(Video).where(col(Video.id) == video_id).values(duration=630.0))
    db.commit()

    def processor(
        _path: Path,
        _corners: list[list[float]],
        _dir: Path,
        on_progress: ProgressFn,
        log: LogFn,
    ) -> run_worker.Detection:
        log("prepared: 1280x720, 30.00 fps, 10.5 min (downscale 0s)")
        for i in range(1, 101):  # 100 progress callbacks...
            on_progress(i / 100)
        log("detecting rallies (tracking took 1s)")
        return detection({"start": 0.0, "end": 30.0}, {"start": 40.0, "end": 70.0})

    caplog.set_level("INFO", logger="worker")
    run(engine, processor)
    p = f"[{str(video_id)[:8]}]"
    assert worker_lines(caplog) == [
        f"{p} claimed: match.mp4, 10.5 min, attempt 1/3",
        f"{p} prepared: 1280x720, 30.00 fps, 10.5 min (downscale 0s)",
        # ...but only one line per 10%
        *[f"{p} tracking {n}%" for n in range(10, 101, 10)],
        f"{p} detecting rallies (tracking took 1s)",
        f"{p} done: 2 rallies (1.0 min of play) in 0s, detector v1.1-2026-11",
    ]


def test_failure_logs_the_reason(
    db: Session, engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    """A problem with the video: one line, no traceback. A bug: with traceback."""
    caplog.set_level("INFO", logger="worker")
    for error, wants_traceback in [
        (CourtNotVisible("shuttle inside the court zone in 0.4% of frames"), False),
        (KeyError("rallies"), True),
    ]:
        caplog.clear()
        video_id = make_video(db)

        def processor(*_: Any, error: Exception = error) -> run_worker.Detection:
            raise error

        run(engine, processor)
        failed = [r for r in caplog.records if "failed:" in r.getMessage()]
        assert len(failed) == 1
        reason = fetch(engine, video_id).failure_reason
        assert failed[0].getMessage() == (
            f"[{str(video_id)[:8]}] failed: {reason} ({type(error).__name__}: {error})"
        )
        assert (failed[0].exc_info is not None) is wants_traceback


def test_lost_job_is_logged(
    db: Session, engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    video_id = make_video(db)

    def processor(*_: Any) -> run_worker.Detection:
        with Session(engine) as session:
            session.exec(
                update(Video)
                .where(col(Video.id) == video_id)
                .values(claim_id=uuid.uuid4())
            )
            session.commit()
        return detection()

    caplog.set_level("INFO", logger="worker")
    run(engine, processor)
    assert worker_lines(caplog)[-1] == (
        f"[{str(video_id)[:8]}] lost: results discarded, another worker has this job now"
    )


def test_trajectory_is_moved_into_place_when_done(db: Session, engine: Engine) -> None:
    video_id = make_video(db)
    published = trajectory.trajectory_path(video_id)
    seen_during_processing: list[bool] = []

    def processor(
        _path: Path,
        _corners: list[list[float]],
        work_dir: Path,
        _progress: ProgressFn,
        _log: LogFn,
    ) -> run_worker.Detection:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / trajectory.FILENAME).write_bytes(b"new trajectory")
        seen_during_processing.append(published.exists())
        return detection()

    run(engine, processor)
    assert seen_during_processing == [False]  # only published once finished
    assert published.read_bytes() == b"new trajectory"
    assert not (published.parent / "work" / trajectory.FILENAME).exists()
    assert fetch(engine, video_id).status == "done"


def test_failed_job_keeps_the_old_trajectory(db: Session, engine: Engine) -> None:
    video_id = make_video(db)
    published = trajectory.trajectory_path(video_id)
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_bytes(b"from the last run")

    def processor(
        _path: Path,
        _corners: list[list[float]],
        work_dir: Path,
        _progress: ProgressFn,
        _log: LogFn,
    ) -> run_worker.Detection:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / trajectory.FILENAME).write_bytes(b"half-finished run")
        raise CourtNotVisible("shuttle inside the court zone in 0.0% of frames")

    run(engine, processor)
    assert fetch(engine, video_id).status == "failed"
    assert published.read_bytes() == b"from the last run"


def saved_rallies(engine: Engine, video_id: uuid.UUID) -> list[Rally]:
    with Session(engine) as session:
        return list(
            session.exec(
                select(Rally)
                .where(col(Rally.video_id) == video_id)
                .order_by(col(Rally.start))
            ).all()
        )


def test_saves_detector_version_confidence_and_maybe(
    db: Session, engine: Engine
) -> None:
    video_id = make_video(db)
    run(
        engine,
        lambda *_: detection(
            {"start": 1.0, "end": 5.0, "confidence": 0.97, "is_maybe": False},
            {"start": 9.0, "end": 12.0, "confidence": 0.41, "is_maybe": True},
        ),
    )
    assert fetch(engine, video_id).detector_version == "v1.1-2026-11"
    saved = saved_rallies(engine, video_id)
    assert [(r.confidence, r.is_maybe, r.source) for r in saved] == [
        (0.97, False, "model"),
        (0.41, True, "model"),
    ]
    assert [(r.detected_start, r.detected_end) for r in saved] == [
        (1.0, 5.0),
        (9.0, 12.0),
    ]


def test_v1_rallies_are_saved_as_rules(db: Session, engine: Engine) -> None:
    video_id = make_video(db)
    run(engine, lambda *_: detection({"start": 1.0, "end": 5.0}, version="v1"))
    assert fetch(engine, video_id).detector_version == "v1"
    [rally] = saved_rallies(engine, video_id)
    assert (rally.source, rally.confidence, rally.is_maybe) == ("rules", None, False)


def test_reanalysis_records_the_new_detector(db: Session, engine: Engine) -> None:
    """Old videos keep their rallies and version until re-analyzed."""
    video_id = make_video(db)
    run(engine, lambda *_: detection({"start": 1.0, "end": 5.0}, version="v1"))
    with Session(engine) as session:
        assert crud.queue_video(session=session, video_id=video_id) == "queued"
    assert fetch(engine, video_id).detector_version == "v1"  # still shown while queued
    run(engine, lambda *_: detection({"start": 2.0, "end": 6.0, "is_maybe": True}))
    assert fetch(engine, video_id).detector_version == "v1.1-2026-11"
    assert [(r.start, r.is_maybe) for r in saved_rallies(engine, video_id)] == [
        (2.0, True)
    ]


def test_done_line_counts_maybes(
    db: Session, engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    video_id = make_video(db)
    caplog.set_level("INFO", logger="worker")
    run(
        engine,
        lambda *_: detection(
            {"start": 0.0, "end": 30.0}, {"start": 40.0, "end": 70.0, "is_maybe": True}
        ),
    )
    assert worker_lines(caplog)[-1] == (
        f"[{str(video_id)[:8]}] done: 2 rallies, 1 maybe (1.0 min of play) in 0s, "
        "detector v1.1-2026-11"
    )


def test_processor_without_a_trajectory_still_finishes(
    db: Session, engine: Engine
) -> None:
    video_id = make_video(db)
    run(engine, lambda *_: detection({"start": 1.0, "end": 2.0}))
    assert fetch(engine, video_id).status == "done"
    assert not trajectory.trajectory_path(video_id).exists()
