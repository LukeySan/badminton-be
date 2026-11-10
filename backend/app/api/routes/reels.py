"""
Reels: highlight edits of a video, stored as an edit list (app.models.ReelEdit).
Non-destructive: the original video is never changed, the app plays the edit list.

Access follows the video: its owner, or a superuser. Deleted reels (soft delete)
are 404 everywhere.
"""

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.exceptions import RequestValidationError
from sqlmodel import Session

from app import crud
from app.api.deps import CurrentUser, SessionDep
from app.api.errors import CodedHTTPException
from app.api.routes.videos import get_video_for_user
from app.models import (
    ErrorCode,
    ErrorWithCode,
    Message,
    Reel,
    ReelCreate,
    ReelEdit,
    ReelPublic,
    ReelsPublic,
    ReelUpdate,
    User,
    Video,
)

router = APIRouter(tags=["reels"])

INVALID_EDIT: dict[int | str, dict[str, Any]] = {
    422: {
        "description": "Invalid edit list, including checks against the video: a "
        "rally_id that isn't one of its rallies, or times past its end. Errors have "
        'the usual `loc`, e.g. ["body", "edit", "clips", 2, "trim_end"]'
    }
}


def get_reel_for_user(session: Session, current_user: User, reel_id: uuid.UUID) -> Reel:
    reel = crud.get_reel(session=session, reel_id=reel_id)
    # 404 for other people's reels too, like videos: don't confirm an id exists
    if not reel or (not current_user.is_superuser and reel.owner_id != current_user.id):
        raise HTTPException(status_code=404, detail="Reel not found")
    return reel


def check_edit_against_video(session: Session, video: Video, edit: ReelEdit) -> None:
    """The checks Pydantic can't do alone, because they need the video's rallies
    and duration. Reported in the same 422 format as Pydantic's, all at once."""
    problems = edit.problems(
        duration=video.duration,
        rally_ids=crud.video_rally_ids(session=session, video_id=video.id),
    )
    if problems:
        raise RequestValidationError(
            [
                {"type": "value_error", "loc": ("body", "edit", *loc), "msg": msg}
                for loc, msg in problems
            ]
        )


@router.get("/videos/{id}/reels", response_model=ReelsPublic)
def read_reels(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    A video's reels, most recently saved first.
    """
    video = get_video_for_user(session, current_user, id)
    reels, count = crud.list_reels(
        session=session, video_id=video.id, skip=skip, limit=limit
    )
    return ReelsPublic(data=reels, count=count)  # type: ignore[arg-type]


@router.post(
    "/videos/{id}/reels",
    response_model=ReelPublic,
    status_code=201,
    responses=INVALID_EDIT,
)
def create_reel(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID, reel_in: ReelCreate
) -> Any:
    """
    Create a reel for a video. It belongs to the video's owner.
    """
    video = get_video_for_user(session, current_user, id)
    check_edit_against_video(session, video, reel_in.edit)
    return crud.create_reel(
        session=session, video=video, title=reel_in.title, edit=reel_in.edit
    )


@router.get("/reels/{reel_id}", response_model=ReelPublic)
def read_reel(
    session: SessionDep, current_user: CurrentUser, reel_id: uuid.UUID
) -> Any:
    """
    Get a reel with its edit list.
    """
    return get_reel_for_user(session, current_user, reel_id)


@router.put(
    "/reels/{reel_id}",
    response_model=ReelPublic,
    responses={
        409: {
            "model": ErrorWithCode,
            "description": "code `reel_changed`: saved elsewhere (another tab or "
            "device) since you loaded it, or its clips were re-linked after the "
            "video was re-analyzed. Load it again.",
        },
        **INVALID_EDIT,
    },
)
def update_reel(
    session: SessionDep,
    current_user: CurrentUser,
    reel_id: uuid.UUID,
    reel_in: ReelUpdate,
) -> Any:
    """
    Save a reel: replaces the title and the whole edit list. Send the `updated_at`
    you loaded; if the reel was saved since, nothing changes and you get 409.
    """
    reel = get_reel_for_user(session, current_user, reel_id)
    video = session.get(Video, reel.video_id)
    assert video  # reels go when their video is deleted (database cascade)
    check_edit_against_video(session, video, reel_in.edit)
    if not crud.update_reel(
        session=session,
        reel_id=reel_id,
        title=reel_in.title,
        edit=reel_in.edit,
        expected_updated_at=reel_in.updated_at,
    ):
        session.expire_all()
        if crud.get_reel(session=session, reel_id=reel_id) is None:
            raise HTTPException(status_code=404, detail="Reel not found")
        raise CodedHTTPException(
            409,
            ErrorCode.REEL_CHANGED,
            "This reel was changed somewhere else since you opened it; reload it",
        )
    session.refresh(reel)
    return reel


@router.delete("/reels/{reel_id}", response_model=Message)
def delete_reel(
    session: SessionDep, current_user: CurrentUser, reel_id: uuid.UUID
) -> Any:
    """
    Delete a reel. The video and its rallies are not affected.
    """
    get_reel_for_user(session, current_user, reel_id)
    if not crud.soft_delete_reel(session=session, reel_id=reel_id):
        raise HTTPException(status_code=404, detail="Reel not found")
    return Message(message="Reel deleted successfully")
