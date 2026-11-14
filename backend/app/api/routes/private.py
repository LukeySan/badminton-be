import gzip
import importlib.util
import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import crud, thumbnails, trajectory
from app.api.deps import SessionDep
from app.core.config import settings
from app.core.security import get_password_hash
from app.models import (
    User,
    UserPublic,
    Video,
    VideoPublic,
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
    pipeline. Goes through crud.finish_video like a real worker, so reel clips are
    re-linked the same way.
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
