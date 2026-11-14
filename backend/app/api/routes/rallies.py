"""
Rally corrections: the user fixes what the detector found. Every change is logged in
the `correction` table next to the detector's original bounds (training labels), and
reels that use a changed rally are adjusted in the same transaction. Each endpoint
returns a RallyChange: what was added/updated/deleted, the renumbered list, and the
reels that were adjusted.

Not allowed while the video is queued or processing (re-analysis replaces all
rallies, which would undo the correction).
"""

import uuid
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException

from app import crud
from app.api.deps import CurrentUser, SessionDep
from app.api.routes.videos import get_video_for_user
from app.models import RallyChange, RallyMerge, RallySplit, RallyTimes

router = APIRouter(prefix="/videos/{id}/rallies", tags=["rallies"])

REFUSALS: dict[int | str, dict[str, Any]] = {
    404: {"description": "Video or rally not found"},
    409: {
        "description": "The video is queued/processing, the times overlap another "
        "rally, or (merge) the rallies aren't neighbours"
    },
    422: {"description": "Invalid times (e.g. past the end of the video)"},
}


def run(correction: Callable[[], RallyChange]) -> RallyChange:
    try:
        return correction()
    except crud.CorrectionRefused as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@router.post("", response_model=RallyChange, status_code=201, responses=REFUSALS)
def add_rally(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID, times: RallyTimes
) -> Any:
    """
    Add a rally the detector missed. It must not overlap another rally.
    """
    get_video_for_user(session, current_user, id)
    return run(
        lambda: crud.add_rally(
            session=session,
            video_id=id,
            user_id=current_user.id,
            start=times.start,
            end=times.end,
        )
    )


@router.put("/{rally_id}", response_model=RallyChange, responses=REFUSALS)
def edit_rally(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    rally_id: int,
    times: RallyTimes,
) -> Any:
    """
    Set a rally's start and end. Reel pieces outside the new bounds (plus each
    reel's lead_in/lead_out) are trimmed.
    """
    get_video_for_user(session, current_user, id)
    return run(
        lambda: crud.edit_rally(
            session=session,
            video_id=id,
            user_id=current_user.id,
            rally_id=rally_id,
            start=times.start,
            end=times.end,
        )
    )


@router.delete("/{rally_id}", response_model=RallyChange, responses=REFUSALS)
def delete_rally(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID, rally_id: int
) -> Any:
    """
    Delete a rally (e.g. a false detection). Reel clips of it are dropped.
    """
    get_video_for_user(session, current_user, id)
    return run(
        lambda: crud.delete_rally(
            session=session, video_id=id, user_id=current_user.id, rally_id=rally_id
        )
    )


@router.post("/{rally_id}/confirm", response_model=RallyChange, responses=REFUSALS)
def confirm_rally(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID, rally_id: int
) -> Any:
    """
    "Yes, that's a rally": clears `is_maybe` on a rally the detector was unsure
    about. Logged as a correction (a training label). No-op for other rallies.
    To answer "no", delete the rally.
    """
    get_video_for_user(session, current_user, id)
    return run(
        lambda: crud.confirm_rally(
            session=session, video_id=id, user_id=current_user.id, rally_id=rally_id
        )
    )


@router.post("/{rally_id}/split", response_model=RallyChange, responses=REFUSALS)
def split_rally(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    rally_id: int,
    split: RallySplit,
) -> Any:
    """
    Split a rally in two at `at` seconds. The first part keeps the id; the second
    is a new rally. Reel clips are split the same way.
    """
    get_video_for_user(session, current_user, id)
    return run(
        lambda: crud.split_rally(
            session=session,
            video_id=id,
            user_id=current_user.id,
            rally_id=rally_id,
            at=split.at,
        )
    )


@router.post("/merge", response_model=RallyChange, responses=REFUSALS)
def merge_rallies(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID, merge: RallyMerge
) -> Any:
    """
    Merge two neighbouring rallies into one new rally (new id). Reel clips of
    either point at the new rally; back-to-back clips of the two become one.
    """
    get_video_for_user(session, current_user, id)
    return run(
        lambda: crud.merge_rallies(
            session=session,
            video_id=id,
            user_id=current_user.id,
            rally_ids=merge.rally_ids,
        )
    )
