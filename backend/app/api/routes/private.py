import gzip
import json
import os
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import crud, trajectory
from app.api.deps import SessionDep
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


class PrivateAnalysis(BaseModel):
    rallies: list[tuple[float, float]]
    trajectory: PrivateTrajectory | None = None


@router.post("/videos/{id}/analysis", response_model=VideoPublic)
def set_analysis(id: uuid.UUID, body: PrivateAnalysis, session: SessionDep) -> Any:
    """
    Test fixture: finish a video as the worker would, with these rallies (and
    optionally a shuttle trajectory), without running the ML pipeline. Goes through
    crud.finish_video like a real worker, so reel clips are re-linked the same way.
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
    crud.finish_video(
        session=session, video_id=id, claim_id=claim_id, rallies=body.rallies
    )
    session.refresh(video)
    return crud.get_video_public(session=session, video=video)
