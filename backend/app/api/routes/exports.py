"""
Reel exports: the video's reel rendered into an MP4 by the worker.

    POST /videos/{id}/reel/exports    queue an export of the video's reel
    GET  /videos/{id}/reel/exports    the video's exports, newest first
    GET  /exports/{id}                one export (poll this for status/progress)
    GET  /exports/{id}/download       signed link to the finished file
    GET  /exports/{id}/file?token=    the file, as an attachment
    POST /exports/{id}/retry          queue a failed export again (same snapshot)
    DELETE /exports/{id}              soft delete (also stops a running render)

What's rendered is fixed when the export is requested (the snapshot, see
app/export_snapshot.py): always the reel's "Rallies only" result, whatever view the
editor shows. Files are deleted EXPORT_RETENTION_DAYS (7) after the request by the
purge command (app/purge.py); `expires_at` says when.

Access follows the video: its owner, or a superuser (404 otherwise).
"""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from sqlmodel import Session

from app import crud
from app.api.deps import CurrentUser, SessionDep
from app.api.errors import CodedHTTPException
from app.api.routes.videos import get_video_for_user
from app.core.config import settings
from app.core.security import EXPORT_FILE_SCOPE, create_file_token, verify_file_token
from app.export_snapshot import build_snapshot
from app.models import (
    ErrorCode,
    ErrorWithCode,
    Export,
    ExportCreate,
    ExportDownload,
    ExportPublic,
    ExportsPublic,
    Message,
    ReelEdit,
    User,
    Video,
    export_filename,
)

router = APIRouter(tags=["exports"])

MIN_1080P_SOURCE = 1080  # the source's shorter side, in pixels


def get_export_for_user(
    session: Session, current_user: User, export_id: uuid.UUID
) -> Export:
    export = crud.get_export(session=session, export_id=export_id)
    # 404 for other people's exports too: don't confirm that an id exists
    if not export or (
        not current_user.is_superuser and export.owner_id != current_user.id
    ):
        raise HTTPException(status_code=404, detail="Export not found")
    return export


def check_quality(video: Video, quality: str) -> None:
    if quality != "1080p":
        return
    short_side = min(video.width or 0, video.height or 0)
    if short_side < MIN_1080P_SOURCE:
        raise HTTPException(
            status_code=422,
            detail=f"1080p needs a video of at least 1080p; this one is "
            f"{video.width}x{video.height}. Export in 720p instead.",
        )


@router.post(
    "/videos/{id}/reel/exports",
    response_model=ExportPublic,
    status_code=201,
    responses={
        404: {
            "model": ErrorWithCode,
            "description": "code `reel_missing`: the video has no reel yet. A video "
            "that doesn't exist gives a plain 404 without a code.",
        },
        409: {
            "description": "The reel has no clips, or an export of it is already "
            "queued or processing"
        },
        422: {
            "description": "1080p for a video smaller than 1080p, or more pieces than "
            "one export can take (MAX_EXPORT_SEGMENTS)"
        },
        429: {"description": "Too many of your exports queued or processing"},
    },
)
def create_export(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    export_in: ExportCreate,
) -> Any:
    """
    Export the video's reel to an MP4, rendered by a worker; poll GET /exports/{id}.
    What's rendered is fixed now: editing the reel or its rallies afterwards
    doesn't change this export. Always the "Rallies only" result (the clips' pieces
    and restored breaks), even when the editor is in "Full video" view.
    At most MAX_ACTIVE_EXPORTS_PER_USER (3) queued or processing per user (429).
    """
    video = get_video_for_user(session, current_user, id)
    reel = crud.get_video_reel(session=session, video_id=id)
    if reel is None:
        raise CodedHTTPException(
            404, ErrorCode.REEL_MISSING, "This video has no reel yet"
        )
    check_quality(video, export_in.quality)
    snapshot = build_snapshot(
        ReelEdit.model_validate(reel.edit),
        duration=video.duration,
        format=export_in.format,
    )
    if not any(s.kind == "piece" for s in snapshot.segments):
        raise HTTPException(
            status_code=409, detail="The reel has no clips yet; add some to export it"
        )
    if len(snapshot.segments) > settings.MAX_EXPORT_SEGMENTS:
        raise HTTPException(
            status_code=422,
            detail=f"The reel has {len(snapshot.segments)} pieces and breaks; one "
            f"export can take at most {settings.MAX_EXPORT_SEGMENTS}",
        )
    export, result = crud.create_export(
        session=session,
        video=video,
        reel=reel,
        snapshot=snapshot,
        quality=export_in.quality,
        max_active=settings.MAX_ACTIVE_EXPORTS_PER_USER,
    )
    if result == "already_active":
        raise HTTPException(
            status_code=409,
            detail="This reel is already being exported; wait for that export to "
            "finish",
        )
    if result == "limit_reached":
        n = settings.MAX_ACTIVE_EXPORTS_PER_USER
        raise HTTPException(
            status_code=429,
            detail=f"You already have {n} export{'s' if n != 1 else ''} queued or "
            "processing; try again when one has finished",
        )
    assert export is not None
    return crud.export_public(export)


@router.get("/videos/{id}/reel/exports", response_model=ExportsPublic)
def read_video_exports(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID
) -> Any:
    """
    The video's exports, newest first (not deleted; expired ones until the purge
    removes them).
    """
    get_video_for_user(session, current_user, id)
    exports, count = crud.list_video_exports(session=session, video_id=id)
    return ExportsPublic(data=[crud.export_public(e) for e in exports], count=count)


@router.get("/exports/{export_id}", response_model=ExportPublic)
def read_export(
    session: SessionDep, current_user: CurrentUser, export_id: uuid.UUID
) -> Any:
    """
    One export: status (queued → processing → done | failed), progress (0-1),
    failure reason, size, and when its file will be deleted.
    """
    return crud.export_public(get_export_for_user(session, current_user, export_id))


@router.post(
    "/exports/{export_id}/retry",
    response_model=ExportPublic,
    responses={
        409: {
            "description": "Not failed (queued, processing or done), or another "
            "export of this reel is already queued or processing"
        },
        410: {"description": "Expired: export the reel again instead"},
        429: {"description": "Too many of your exports queued or processing"},
    },
)
def retry_export(
    session: SessionDep, current_user: CurrentUser, export_id: uuid.UUID
) -> Any:
    """
    Queue a failed export again ("Try again"). It renders the same snapshot: what
    the reel was when the export was requested, not what it is now. Keeps its
    created_at and expires_at.
    """
    get_export_for_user(session, current_user, export_id)
    result = crud.retry_export(
        session=session,
        export_id=export_id,
        max_active=settings.MAX_ACTIVE_EXPORTS_PER_USER,
    )
    if result == "not_failed":
        raise HTTPException(
            status_code=409, detail="Only a failed export can be tried again"
        )
    if result == "expired":
        raise HTTPException(
            status_code=410,
            detail="This export has expired; export the reel again instead",
        )
    if result == "already_active":
        raise HTTPException(
            status_code=409,
            detail="This reel is already being exported; wait for that export to "
            "finish",
        )
    if result == "limit_reached":
        n = settings.MAX_ACTIVE_EXPORTS_PER_USER
        raise HTTPException(
            status_code=429,
            detail=f"You already have {n} export{'s' if n != 1 else ''} queued or "
            "processing; try again when one has finished",
        )
    export = crud.get_export(session=session, export_id=export_id)
    assert export is not None
    session.refresh(export)
    return crud.export_public(export)


@router.get(
    "/exports/{export_id}/download",
    response_model=ExportDownload,
    responses={
        409: {"description": "Not finished yet (or it failed)"},
        410: {"description": "Expired: the file has been or is about to be deleted"},
    },
)
def read_export_download(
    session: SessionDep, current_user: CurrentUser, export_id: uuid.UUID
) -> Any:
    """
    A temporary link to download the finished MP4 (saved as `filename`). The link
    expires after 12 hours, or when the export does.
    """
    export = get_export_for_user(session, current_user, export_id)
    if export.status != "done" or export.output_path is None:
        raise HTTPException(
            status_code=409, detail=f"The export is {export.status}, not ready yet"
        )
    expires_at = crud.export_expires_at(export)
    if expires_at <= datetime.now(UTC):
        raise HTTPException(status_code=410, detail="This export has expired")
    token, link_expires = create_file_token(
        export.id, scope=EXPORT_FILE_SCOPE, not_after=expires_at
    )
    return ExportDownload(
        url=f"{settings.API_V1_STR}/exports/{export.id}/file?token={token}",
        filename=export_filename(export.title),
        expires_at=link_expires,
    )


@router.get(
    "/exports/{export_id}/file",
    response_class=FileResponse,
    responses={
        200: {"content": {"video/mp4": {}}},
        403: {"description": "Invalid or expired link"},
    },
)
def read_export_file(
    session: SessionDep,
    export_id: uuid.UUID,
    token: Annotated[str, Query(description="From GET /exports/{id}/download")],
) -> FileResponse:
    """
    The exported MP4, as an attachment (Content-Disposition), with Range support.
    """
    if not verify_file_token(token, export_id, scope=EXPORT_FILE_SCOPE):
        raise HTTPException(status_code=403, detail="Invalid or expired file link")
    export = crud.get_export(session=session, export_id=export_id)
    if export is None or export.output_path is None:
        raise HTTPException(status_code=404, detail="Export not found")
    path = settings.STORAGE_DIR / export.output_path
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Export not found")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=export_filename(export.title),
        content_disposition_type="attachment",
    )


@router.delete("/exports/{export_id}", response_model=Message)
def delete_export(
    session: SessionDep, current_user: CurrentUser, export_id: uuid.UUID
) -> Any:
    """
    Delete an export (soft) and its file. A render in progress stops.
    """
    get_export_for_user(session, current_user, export_id)
    path = crud.soft_delete_export(session=session, export_id=export_id)
    if path is None:  # deleted by another request meanwhile
        raise HTTPException(status_code=404, detail="Export not found")
    if path:  # after the commit: at worst an orphan file, never a dangling row
        (settings.STORAGE_DIR / path).unlink(missing_ok=True)
    return Message(message="Export deleted successfully")
