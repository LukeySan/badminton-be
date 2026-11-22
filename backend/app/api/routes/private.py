import gzip
import importlib
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import crud, thumbnails, trajectory
from app.api.deps import SessionDep
from app.core.config import settings
from app.core.security import get_password_hash
from app.models import (
    EXPORT_SIZES,
    Export,
    ExportFailureReason,
    ExportPublic,
    User,
    UserPublic,
    Video,
    VideoPublic,
    get_datetime_utc,
)

router = APIRouter(tags=["private"], prefix="/private")


class PrivateUserCreate(BaseModel):
    email: str
    password: str
    full_name: str
    is_verified: bool = False


@router.post("/users/", response_model=UserPublic)
def create_user(user_in: PrivateUserCreate, session: SessionDep) -> Any:
    """
    Create a new user.
    """

    user = User(
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=get_password_hash(user_in.password),
    )

    session.add(user)
    session.commit()

    return user


class PrivateTrajectory(BaseModel):
    """The worker's trajectory file contents (see app/trajectory.py)."""

    fps: float = Field(gt=0)
    visible: list[int]
    x: list[float | None]
    y: list[float | None]


class PrivateRally(BaseModel):
    start: float
    end: float
    confidence: float | None = None
    is_maybe: bool = False


class PrivateAnalysis(BaseModel):
    rallies: list[tuple[float, float] | PrivateRally] = Field(
        description="[start, end] pairs, or objects with confidence / is_maybe"
    )
    detector_version: str | None = "test-fixture"
    trajectory: PrivateTrajectory | None = None
    thumbnails: bool = Field(
        default=False,
        description="Make the thumbnail strip from the uploaded file (needs ffmpeg)",
    )


def _make_thumbnails(video: Video) -> None:
    """The worker's thumbnail strip, from the uploaded file. Loaded from worker/ by
    path: the API doesn't depend on the worker package (dev-only fixture)."""
    source = Path(__file__).resolve().parents[4] / "worker" / "thumbnail_strip.py"
    spec = importlib.util.spec_from_file_location("thumbnail_strip", source)
    assert spec and spec.loader
    strip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strip)
    out = thumbnails.thumbnails_dir(video.id)
    out.parent.mkdir(parents=True, exist_ok=True)
    strip.generate_thumbnails(str(settings.STORAGE_DIR / video.storage_path), str(out))


@router.post("/videos/{id}/analysis", response_model=VideoPublic)
def set_analysis(id: uuid.UUID, body: PrivateAnalysis, session: SessionDep) -> Any:
    """
    Test fixture: finish a video as the worker would, with these rallies (and
    optionally a shuttle trajectory and the thumbnail strip), without running the ML
    pipeline. Goes through crud.finish_video like a real worker.
    """
    video = session.get(Video, id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    # Pretend a worker claimed it: finish_video only writes to a claimed video.
    claim_id = uuid.uuid4()
    video.status = "processing"
    video.claim_id = claim_id
    session.add(video)
    session.commit()
    if body.trajectory is not None:
        path = trajectory.trajectory_path(id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as fh:
            json.dump({"version": 1, **body.trajectory.model_dump()}, fh)
        os.replace(tmp, path)
    if body.thumbnails:
        _make_thumbnails(video)
    rallies = [
        crud.DetectedRally(r.start, r.end, r.confidence, r.is_maybe)
        if isinstance(r, PrivateRally)
        else crud.DetectedRally(*r)
        for r in body.rallies
    ]
    crud.finish_video(
        session=session,
        video_id=id,
        claim_id=claim_id,
        rallies=rallies,
        detector_version=body.detector_version,
    )
    session.refresh(video)
    return crud.get_video_public(session=session, video=video)


# --- Reel exports (tests drive an export without a running worker) ----------------

WORKER_DIR = Path(__file__).resolve().parents[4] / "worker"


class PrivateExportState(BaseModel):
    status: Literal["queued", "processing", "failed"]
    progress: float = Field(default=0.0, ge=0, le=1)
    failure_reason: ExportFailureReason = ExportFailureReason.UNKNOWN


def _private_export(session: SessionDep, id: uuid.UUID) -> Export:
    export = session.get(Export, id)
    if export is None or export.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Export not found")
    return export


@router.post("/exports/{id}/state", response_model=ExportPublic)
def set_export_state(
    id: uuid.UUID, body: PrivateExportState, session: SessionDep
) -> Any:
    """
    Test fixture: put an export where a worker would (rendering at some progress,
    or failed with a reason), without rendering. Takes a new claim, so a real
    worker that has it stops writing to it (its writes check the claim).
    """
    export = _private_export(session, id)
    export.status = body.status
    export.progress = body.progress if body.status == "processing" else 0.0
    export.claim_id = uuid.uuid4() if body.status == "processing" else None
    export.failure_reason = (
        body.failure_reason.value if body.status == "failed" else None
    )
    export.error = "test fixture" if body.status == "failed" else None
    export.updated_at = get_datetime_utc()
    session.add(export)
    session.commit()
    session.refresh(export)
    return crud.export_public(export)


def _reel_export_module() -> Any:
    """The worker's renderer, loaded from worker/ (the API doesn't depend on the
    worker package; dev-only fixture). Its own imports are top-level worker modules."""
    if str(WORKER_DIR) not in sys.path:
        sys.path.append(str(WORKER_DIR))
    return importlib.import_module("reel_export")


@router.post("/exports/{id}/render", response_model=ExportPublic)
def render_export_now(id: uuid.UUID, session: SessionDep) -> Any:
    """
    Test fixture: render the export now, inside this request, with the worker's
    renderer (ffmpeg; a few seconds for a short clip), and finish it like a worker:
    a real MP4 behind GET /exports/{id}/download.
    """
    export = _private_export(session, id)
    video = session.get(Video, export.video_id)
    assert video is not None
    claim_id = uuid.uuid4()
    export.status = "processing"
    export.claim_id = claim_id
    export.failure_reason = None
    export.error = None
    session.add(export)
    session.commit()
    width, height = EXPORT_SIZES[(export.format, export.quality)]
    relative = f"{video.id}/exports/{export.id}.mp4"
    result = _reel_export_module().render_export(
        str(settings.STORAGE_DIR / video.storage_path),
        export.snapshot,
        width,
        height,
        str(settings.STORAGE_DIR / relative),
        log=lambda _line: None,
    )
    crud.finish_export(
        session=session,
        export_id=export.id,
        claim_id=claim_id,
        output_path=relative,
        size_bytes=int(result["size_bytes"]),
        duration=float(result["duration"]),
    )
    session.refresh(export)
    return crud.export_public(export)
