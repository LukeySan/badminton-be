"""Thumbnail strips: worker/thumbnail_strip.py (with real ffmpeg), publishing them
from the worker loop, and worker/backfill_thumbnails.py."""

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, col, update

from app import thumbnails
from app.core.config import settings
from app.models import Video
from tests.crud.test_video_jobs import (  # noqa: F401  (empty_videos is a fixture)
    empty_videos,
    fetch,
    make_video,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "worker"))
import backfill_thumbnails  # noqa: E402
import run_worker  # noqa: E402
import thumbnail_strip  # noqa: E402
from job_errors import CourtNotVisible  # noqa: E402


def make_clip(path: Path, duration: float, size: str = "320x240") -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc=duration={duration}:size={size}:rate=30",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )  # fmt: skip
    return path


def image_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    w, h = out.strip().split(",")
    return int(w), int(h)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_clip(tmp_path_factory.mktemp("clips") / "clip.mp4", 3.5)


# --- the strip ------------------------------------------------------------------


def test_strip_and_index(clip: Path, tmp_path: Path) -> None:
    out = thumbnail_strip.generate_thumbnails(str(clip), str(tmp_path / "thumbnails"))
    index = json.loads((Path(out) / "index.json").read_text())
    assert index == {
        "version": 1,
        "interval": 1.0,
        "count": 4,  # 3.5 s: tiles at 0, 1, 2, 3 s
        "tile_width": 160,
        "tile_height": 120,  # 320x240 keeps its shape
        "columns": 10,
        "rows": 10,
        "sheets": ["sheet_000.jpg"],
    }
    assert image_size(Path(out) / "sheet_000.jpg") == (1600, 1200)
    assert not Path(out + ".tmp").exists()


def test_long_video_gets_several_sheets(tmp_path: Path) -> None:
    long_clip = make_clip(tmp_path / "long.mp4", 205, size="64x36")
    out = Path(thumbnail_strip.generate_thumbnails(str(long_clip), str(tmp_path / "t")))
    index = json.loads((out / "index.json").read_text())
    assert index["count"] == 205
    assert index["sheets"] == ["sheet_000.jpg", "sheet_001.jpg", "sheet_002.jpg"]
    assert (index["tile_width"], index["tile_height"]) == (160, 90)


def test_portrait_video(tmp_path: Path) -> None:
    portrait = make_clip(tmp_path / "p.mp4", 1, size="360x640")
    out = Path(thumbnail_strip.generate_thumbnails(str(portrait), str(tmp_path / "t")))
    index = json.loads((out / "index.json").read_text())
    assert (index["tile_width"], index["tile_height"]) == (160, 284)


@pytest.mark.parametrize("duration", [0.5, 2.98, 3.0, 3.4, 3.5, 3.6])
def test_tile_count_matches_ffmpeg(tmp_path: Path, duration: float) -> None:
    """The index's count must be the number of tiles ffmpeg actually makes, or the
    app would show black tiles at the end (or miss the last ones)."""
    video = make_clip(tmp_path / "v.mp4", duration)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", "fps=1",
         str(tmp_path / "f_%03d.jpg")],
        check=True,
    )  # fmt: skip
    made = len(list(tmp_path.glob("f_*.jpg")))
    assert thumbnail_strip.tile_count(duration) == made


def test_regenerating_replaces_the_old_strip(clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "thumbnails"
    out.mkdir()
    (out / "sheet_007.jpg").write_bytes(b"old")
    thumbnail_strip.generate_thumbnails(str(clip), str(out))
    assert sorted(p.name for p in out.iterdir()) == ["index.json", "sheet_000.jpg"]
    assert not (tmp_path / "thumbnails.old").exists()


def test_mkv_and_webm_have_no_stream_duration(clip: Path, tmp_path: Path) -> None:
    """Matroska (and WebM, which is Matroska) only store the file's duration."""
    mkv = tmp_path / "clip.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(clip), "-c", "copy", str(mkv)], check=True
    )
    out = thumbnail_strip.generate_thumbnails(str(mkv), str(tmp_path / "t"))
    assert json.loads((Path(out) / "index.json").read_text())["count"] == 4


def test_unreadable_video_leaves_nothing(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(subprocess.CalledProcessError):
        thumbnail_strip.generate_thumbnails(str(junk), str(tmp_path / "t"))
    assert not (tmp_path / "t").exists()


# --- published by the worker loop ------------------------------------------------


def processor_writing_strip(fail: bool = False) -> run_worker.Processor:
    def processor(
        _path: Path, _corners: Any, work_dir: Path, _progress: Any, _log: Any
    ) -> list[dict[str, Any]]:
        strip = work_dir / "thumbnails"
        strip.mkdir(parents=True, exist_ok=True)
        (strip / "index.json").write_text('{"sheets": ["sheet_000.jpg"]}')
        (strip / "sheet_000.jpg").write_bytes(b"new")
        if fail:
            raise CourtNotVisible("shuttle inside the court zone in 0.0% of frames")
        return []

    return processor


def test_worker_publishes_the_strip(db: Session, engine: Engine) -> None:
    video_id = make_video(db)
    old = thumbnails.thumbnails_dir(video_id)
    old.mkdir(parents=True)
    (old / "sheet_009.jpg").write_bytes(b"from an older run")
    run_worker.process_one(processor_writing_strip(), "w", engine=engine)
    assert fetch(engine, video_id).status == "done"
    assert sorted(p.name for p in old.iterdir()) == ["index.json", "sheet_000.jpg"]
    assert thumbnails.read_index(video_id) == {"sheets": ["sheet_000.jpg"]}


def test_failed_run_keeps_the_old_strip(db: Session, engine: Engine) -> None:
    video_id = make_video(db)
    old = thumbnails.thumbnails_dir(video_id)
    old.mkdir(parents=True)
    (old / "index.json").write_text('{"sheets": []}')
    run_worker.process_one(processor_writing_strip(fail=True), "w", engine=engine)
    assert fetch(engine, video_id).status == "failed"
    assert thumbnails.read_index(video_id) == {"sheets": []}


# --- backfill ----------------------------------------------------------------------


def stored_video(db: Session, clip: Path, status: str = "done") -> uuid.UUID:
    video_id = make_video(db, status=status)
    folder = settings.STORAGE_DIR / str(video_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "original.mp4").write_bytes(clip.read_bytes())
    db.exec(
        update(Video)
        .where(col(Video.id) == video_id)
        .values(storage_path=f"{video_id}/original.mp4")
    )
    db.commit()
    return video_id


def test_backfill(db: Session, engine: Engine, clip: Path) -> None:
    missing = stored_video(db, clip)
    has_strip = stored_video(db, clip)
    thumbnails.thumbnails_dir(has_strip).mkdir()
    (thumbnails.thumbnails_dir(has_strip) / "index.json").write_text("{}")
    queued = stored_video(db, clip, status="queued")  # the worker will make its strip
    no_file = make_video(db, status="done")  # storage_path points nowhere

    preview = backfill_thumbnails.backfill(engine, dry_run=True)
    assert set(preview["made"]) == {missing, no_file}
    assert thumbnails.read_index(missing) is None  # a dry run makes nothing

    outcome = backfill_thumbnails.backfill(engine)
    assert outcome == {"made": [missing], "skipped": [has_strip], "failed": [no_file]}
    index = thumbnails.read_index(missing)
    assert index and index["count"] == 4
    assert thumbnails.read_index(queued) is None
    # Running again: nothing left to do
    assert backfill_thumbnails.backfill(engine, video_ids=[missing])["skipped"] == [
        missing
    ]


def test_backfill_prefers_the_analysis_copy_and_can_force(
    db: Session, engine: Engine, clip: Path
) -> None:
    video_id = stored_video(db, clip)
    thumbnails.thumbnails_dir(video_id).mkdir()
    (thumbnails.thumbnails_dir(video_id) / "index.json").write_text("{}")
    work = settings.STORAGE_DIR / str(video_id) / "work"
    work.mkdir()
    make_clip(work / "video_720.mp4", 1.0)  # differs from the 3.5 s original
    outcome = backfill_thumbnails.backfill(engine, video_ids=[video_id], force=True)
    assert outcome["made"] == [video_id]
    index = thumbnails.read_index(video_id)
    assert index and index["count"] == 1  # made from the 1 s analysis copy


def test_backfill_command_exit_code(
    db: Session, engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = make_video(db, status="done")
    monkeypatch.setattr(sys, "argv", ["backfill", "--video", str(broken)])
    with pytest.raises(SystemExit) as exit_info:
        backfill_thumbnails.main(engine)
    assert exit_info.value.code == 1  # a failure is reported, not hidden
