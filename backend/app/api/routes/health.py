from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app import crud
from app.api.deps import SessionDep
from app.models import Health

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=Health,
    responses={503: {"model": Health, "description": "Database unreachable"}},
)
def health(session: SessionDep) -> Any:
    """
    For monitoring, no login needed: is the database reachable, and is the queue
    moving? Queued videos piling up with nothing processing means no worker is
    running; a growing oldest_queued_seconds means the workers can't keep up.
    """
    try:
        queued, processing, oldest = crud.queue_stats(session=session)
    except SQLAlchemyError:
        return JSONResponse(
            status_code=503,
            content=Health(status="error", database="unreachable").model_dump(),
        )
    return Health(
        status="ok",
        database="ok",
        queued=queued,
        processing=processing,
        oldest_queued_seconds=None if oldest is None else round(oldest, 1),
    )
