"""Exports in the worker loop (worker/run_worker.py): claimed before videos, rendered
with real ffmpeg from the original upload, and failures that fail the export, not
the worker."""

import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, col, update

from app import crud
from app.core.config import settings
from app.models import Export, ExportFailureReason, Video
from tests.crud.test_exports import fetch_export, make_export, set_export_age
from tests.crud.test_video_jobs import (  # noqa: F401  (empty_videos is a fixture)
    empty_videos,
    fetch,
    make_video,
    set_age,
)
from tests.worker.test_run_worker import detection, serve, stop_when_idle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "worker"))
import reel_export  # noqa: E402
import run_worker  # noqa: E402

FAST_BEAT = 0.05


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 4 s 640x360 clip with sound."""
    out = tmp_path_factory.mktemp("export-src") / "original.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=4",
         "-f", "lavfi", "-i", "sine=f=440:d=4", "-shortest",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)],
        check=True,
    )  # fmt: skip
    return out


def video_with_file(db: Session, source: Path, status: str = "done") -> uuid.UUID:
    """A video row whose original is a real file in STORAGE_DIR."""
    video_id = make_video(db, status=status)
    relative = f"{video_id}/original.mp4"
    (settings.STORAGE_DIR / str(video_id)).mkdir(parents=True, exist_ok=True)
    shutil.copy(source, settings.STORAGE_DIR / relative)
    db.exec(
        update(Video)
        .where(col(Video.id) == video_id)
        .values(storage_path=relative, duration=4.0, width=640, height=360)
    )
    db.commit()
    return video_id


def run(
    engine: Engine, renderer: run_worker.ExportRenderer = run_worker.render_export
) -> bool:
    def no_videos(*_: Any) -> run_worker.Detection:
        raise AssertionError("no video job expected")

    return run_worker.process_one(
        no_videos, "test-worker", engine=engine, heartbeat_seconds=FAST_BEAT,
        renderer=renderer,
    )  # fmt: skip


def test_export_is_rendered_and_saved(
    db: Session, engine: Engine, source: Path
) -> None:
    export = make_export(db, video_with_file(db, source))  # pieces 1-3 s
    assert run(engine) is True

    done = fetch_export(engine, export.id)
    assert (done.status, done.progress, done.claim_id) == ("done", 1.0, None)
    assert done.output_path == f"{export.video_id}/exports/{export.id}.mp4"
    path = settings.STORAGE_DIR / done.output_path
    assert done.size_bytes == path.stat().st_size
    assert done.duration is not None and abs(done.duration - 2.0) < 0.1
    info = reel_export.probe(str(path))
    assert info["has_audio"]
    leftovers = sorted(p.name for p in path.parent.iterdir())
    assert leftovers == [path.name]  # no .part file or caption folder


def test_exports_are_claimed_before_videos(
    db: Session, engine: Engine, source: Path
) -> None:
    """A video queued long before still waits: exports are short."""
    waiting = make_video(db, age=timedelta(hours=1))
    export = make_export(db, video_with_file(db, source))
    assert run(engine) is True
    assert fetch_export(engine, export.id).status == "done"
    assert fetch(engine, waiting).status == "queued"


def test_failing_ffmpeg_fails_the_export_not_the_worker(
    db: Session,
    engine: Engine,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ffmpeg exits with an error on the first export; the loop records it and goes
    on to the next export and then the video."""
    broken = make_export(db, video_with_file(db, source))
    set_export_age(db, broken.id, timedelta(minutes=1))  # claimed first
    fine = make_export(db, video_with_file(db, source))
    video_id = make_video(db)

    def renderer(*args: Any) -> reel_export.RenderResult:
        if args[4].stem == str(broken.id):  # the output file is named by export id
            with monkeypatch.context() as m:
                m.setattr(reel_export, "VIDEO_CODEC", ["-c:v", "no_such_encoder"])
                return run_worker.render_export(*args)
        return run_worker.render_export(*args)

    caplog.set_level("INFO", logger="worker")
    serve(engine, lambda *_: detection({"start": 1.0, "end": 2.0}), renderer=renderer,
          wait=stop_when_idle())  # fmt: skip

    failed = fetch_export(engine, broken.id)
    assert (failed.status, failed.failure_reason, failed.claim_id) == (
        "failed",
        ExportFailureReason.UNKNOWN,
        None,
    )
    assert failed.error and failed.error.startswith("RenderFailed: ffmpeg exited with")
    assert "no_such_encoder" in failed.error
    assert not (
        settings.STORAGE_DIR / str(broken.video_id) / "exports" / f"{broken.id}.mp4"
    ).exists()
    assert fetch_export(engine, fine.id).status == "done"
    assert fetch(engine, video_id).status == "done"
    assert any("failed: Something went wrong while exporting" in r.getMessage()
               and r.exc_info for r in caplog.records)  # fmt: skip


def test_missing_original_is_a_source_problem(
    db: Session, engine: Engine, source: Path
) -> None:
    video_id = video_with_file(db, source)
    export = make_export(db, video_id)
    (settings.STORAGE_DIR / str(video_id) / "original.mp4").unlink()
    assert run(engine) is True
    failed = fetch_export(engine, export.id)
    assert failed.status == "failed"
    assert failed.failure_reason == ExportFailureReason.SOURCE_UNREADABLE
    assert failed.error and failed.error.startswith("SourceUnreadable:")


def slow_renderer(
    during: Callable[[], None],
) -> run_worker.ExportRenderer:
    """Writes a partial file, runs `during`, then keeps reporting progress for a
    while (as ffmpeg does), so the heartbeat can notice a lost job."""

    def render(
        _source: Path,
        _snapshot: dict[str, Any],
        _w: int,
        _h: int,
        out_path: Path,
        on_progress: Callable[[float], None],
        _log: Callable[[str], None],
    ) -> reel_export.RenderResult:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"partial")
        during()
        for i in range(40):
            time.sleep(FAST_BEAT / 2)
            on_progress(i / 40)
        return {"duration": 2.0, "size_bytes": 7}

    return render


def test_deleting_the_export_stops_the_render(
    db: Session, engine: Engine, source: Path, caplog: pytest.LogCaptureFixture
) -> None:
    export = make_export(db, video_with_file(db, source))

    def delete_it() -> None:
        with Session(engine) as session:
            crud.soft_delete_export(session=session, export_id=export.id)

    caplog.set_level("INFO", logger="worker")
    assert run(engine, slow_renderer(delete_it)) is True
    row = fetch_export(engine, export.id)
    assert row.deleted_at is not None and row.output_path is None
    out = settings.STORAGE_DIR / str(export.video_id) / "exports" / f"{export.id}.mp4"
    assert not out.exists()
    assert (
        caplog.records[-1]
        .getMessage()
        .endswith("lost: deleted, or another worker has this export now; stopped")
    )


def test_ctrl_c_puts_the_export_back(db: Session, engine: Engine, source: Path) -> None:
    export = make_export(db, video_with_file(db, source))

    def interrupt() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(engine, slow_renderer(interrupt))
    back = fetch_export(engine, export.id)
    assert (back.status, back.attempts, back.claim_id) == ("queued", 0, None)


def test_stale_export_is_rescued(db: Session, engine: Engine, source: Path) -> None:
    export = make_export(db, video_with_file(db, source))
    db.exec(
        update(Export)
        .where(col(Export.id) == export.id)
        .values(status="processing", attempts=1, claim_id=uuid.uuid4())
    )
    db.commit()
    set_export_age(db, export.id, timedelta(minutes=10))
    assert run(engine) is True
    done = fetch_export(engine, export.id)
    assert (done.status, done.attempts) == ("done", 2)


def test_result_saved_too_late_is_discarded(
    db: Session, engine: Engine, source: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Deleted after the last progress update: finish_export refuses, the file goes."""
    export = make_export(db, video_with_file(db, source))

    def render(*args: Any) -> reel_export.RenderResult:
        result = run_worker.render_export(*args)
        with Session(engine) as session:
            crud.soft_delete_export(session=session, export_id=export.id)
        return result

    caplog.set_level("INFO", logger="worker")
    assert run(engine, render) is True
    out = settings.STORAGE_DIR / str(export.video_id) / "exports" / f"{export.id}.mp4"
    assert not out.exists()
    assert (
        caplog.records[-1]
        .getMessage()
        .endswith("lost: export discarded (deleted, or another worker has it now)")
    )


def test_worker_refuses_to_start_without_text_layout(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(run_worker.caption_image, "check_text_layout", lambda: False)
    monkeypatch.setattr(sys, "argv", ["run_worker.py"])
    with pytest.raises(SystemExit) as stopped:
        run_worker.main()
    assert stopped.value.code == 1
    assert "brew install fribidi" in caplog.text
