"""
Reels: highlight edits of a video, stored as an edit list (app.models.ReelEdit).
Non-destructive: the original video is never changed, the app plays the edit list.

Beta: one reel per video, at GET/PUT /videos/{id}/reel. The older /videos/{id}/reels
and /reels/{reel_id} endpoints are deprecated and stay only until the frontend has
switched; then remove them (see CLAUDE.md).

Access follows the video: its owner, or a superuser. Deleted reels (soft delete)
are 404 everywhere.
"""

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Response
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
    ReelSave,
    ReelsPublic,
    ReelUpdate,
    User,
    Video,
)

router = APIRouter(tags=["reels"])

INVALID_EDIT: dict[int | str, dict[str, Any]] = {
    422: {
        "description": "Invalid edit list, including checks against the video: a "
        "rally_id that isn't one of its rallies, pieces outside their rally's bounds "
        "(+ lead_in/lead_out), times past its end. Errors have the usual `loc`, e.g. "
        '["body", "edit", "clips", 2, "pieces", 0, "end"]'
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
    and duration. Reported in the same 422 format as Pydantic's, all at once.
    Locks the video row (shared) first, so rallies can't change between this check
    and the save (see crud.lock_video_for_reel_save)."""
    crud.lock_video_for_reel_save(session=session, video_id=video.id)
    problems = edit.problems(
        duration=video.duration,
        rallies=crud.video_rally_bounds(session=session, video_id=video.id),
    )
    if problems:
        session.rollback()
        raise RequestValidationError(
            [
                {"type": "value_error", "loc": ("body", "edit", *loc), "msg": msg}
                for loc, msg in problems
            ]
        )


REEL_CHANGED = {
    "model": ErrorWithCode,
    "description": "code `reel_changed`: saved, created or deleted elsewhere (another "
    "tab or device) since you loaded it, or adjusted to a rally correction or "
    "re-analysis. Load it again.",
}


def reel_changed() -> CodedHTTPException:
    return CodedHTTPException(
        409,
        ErrorCode.REEL_CHANGED,
        "This reel was changed somewhere else since you opened it; reload it",
    )


# --- One reel per video (beta) --------------------------------------------------


@router.get(
    "/videos/{id}/reel",
    response_model=ReelPublic,
    responses={
        404: {
            "model": ErrorWithCode,
            "description": "code `reel_missing`: the video has no reel yet (create it "
            "with PUT). A video that doesn't exist gives a plain 404 without a code.",
        }
    },
)
def read_video_reel(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID
) -> Any:
    """
    The video's reel with its edit list.
    """
    get_video_for_user(session, current_user, id)
    reel = crud.get_video_reel(session=session, video_id=id)
    if reel is None:
        raise CodedHTTPException(
            404, ErrorCode.REEL_MISSING, "This video has no reel yet"
        )
    return reel


@router.put(
    "/videos/{id}/reel",
    response_model=ReelPublic,
    responses={
        201: {"model": ReelPublic, "description": "Created (updated_at was null)"},
        409: REEL_CHANGED,
        **INVALID_EDIT,
    },
)
def save_video_reel(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    reel_in: ReelSave,
    response: Response,
) -> Any:
    """
    Create or save the video's reel: replaces the title and the whole edit list
    (clips without pieces are dropped). Send `updated_at: null` to create it (201),
    or the `updated_at` you loaded to update it (200). If it isn't as you last saw it
    (it now exists, changed, or was deleted), nothing is saved: 409, reload.
    The reel belongs to the video's owner, also when a superuser saves it.
    """
    video = get_video_for_user(session, current_user, id)
    check_edit_against_video(session, video, reel_in.edit)
    reel, created = crud.save_video_reel(
        session=session,
        video=video,
        title=reel_in.title,
        edit=reel_in.edit,
        expected_updated_at=reel_in.updated_at,
    )
    if reel is None:
        raise reel_changed()
    if created:
        response.status_code = 201
    return reel


# --- Deprecated: several reels per video, by reel id -----------------------------
# Kept until the frontend uses /videos/{id}/reel; then delete these routes, their
# tests in test_reels.py, ReelCreate/ReelUpdate/ReelsPublic and crud.list_reels.


@router.get("/videos/{id}/reels", response_model=ReelsPublic, deprecated=True)
def read_reels(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Deprecated: use GET /videos/{id}/reel. The video's reel (at most one now).
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
    responses={200: {"model": ReelPublic, "description": "The video's existing reel"},
               **INVALID_EDIT},
    deprecated=True,
)  # fmt: skip
def create_reel(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    reel_in: ReelCreate,
    response: Response,
) -> Any:
    """
    Deprecated: use PUT /videos/{id}/reel. Get-or-create: creates the video's reel
    (201), or, if it already has one, returns that one unchanged (200; the body is
    ignored), so the old "make a highlight" flow opens the existing reel.
    """
    video = get_video_for_user(session, current_user, id)
    existing = crud.get_video_reel(session=session, video_id=id)
    if existing is None:
        check_edit_against_video(session, video, reel_in.edit)
        created = crud.create_reel(
            session=session, video=video, title=reel_in.title, edit=reel_in.edit
        )
        if created is not None:
            return created
        # Created by another request a moment ago
        existing = crud.get_video_reel(session=session, video_id=id)
        assert existing is not None
    response.status_code = 200
    return existing


@router.get("/reels/{reel_id}", response_model=ReelPublic, deprecated=True)
def read_reel(
    session: SessionDep, current_user: CurrentUser, reel_id: uuid.UUID
) -> Any:
    """
    Deprecated: use GET /videos/{id}/reel. Get a reel with its edit list.
    """
    return get_reel_for_user(session, current_user, reel_id)


@router.put(
    "/reels/{reel_id}",
    response_model=ReelPublic,
    responses={409: REEL_CHANGED, **INVALID_EDIT},
    deprecated=True,
)
def update_reel(
    session: SessionDep,
    current_user: CurrentUser,
    reel_id: uuid.UUID,
    reel_in: ReelUpdate,
) -> Any:
    """
    Deprecated: use PUT /videos/{id}/reel. Save a reel: replaces the title and the
    whole edit list. Send the `updated_at` you loaded; 409 if it was saved since.
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
        raise reel_changed()
    session.refresh(reel)
    return reel


@router.delete("/reels/{reel_id}", response_model=Message, deprecated=True)
def delete_reel(
    session: SessionDep, current_user: CurrentUser, reel_id: uuid.UUID
) -> Any:
    """
    Deprecated (no replacement yet). Delete a reel (soft); the video and its
    rallies are not affected. The video can then get a new reel.
    """
    get_reel_for_user(session, current_user, reel_id)
    if not crud.soft_delete_reel(session=session, reel_id=reel_id):
        raise HTTPException(status_code=404, detail="Reel not found")
    return Message(message="Reel deleted successfully")
