import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from sqlmodel import Session

from app import crud, media
from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.core.security import create_file_token, verify_file_token
from app.models import (
    CourtUpdate,
    FileUrl,
    Message,
    RalliesPublic,
    User,
    Video,
    VideoPublic,
    VideosPublic,
)

router = APIRouter(prefix="/videos", tags=["videos"])

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
CHUNK = 1024 * 1024

# Handlers are plain `def` (not async): FastAPI runs them in a thread pool, so slow
# file copies and ffmpeg calls don't block other requests.


def get_video_for_user(
    session: Session, current_user: User, video_id: uuid.UUID
) -> Video:
    video = session.get(Video, video_id)
    # 404 rather than 403 for other people's videos: don't confirm that an id exists
    if not video or (
        not current_user.is_superuser and video.owner_id != current_user.id
    ):
        raise HTTPException(status_code=404, detail="Video not found")
    return video


@router.get(
    "/",
    response_model=VideosPublic,
    responses={403: {"description": "all=true without superuser rights"}},
)
def read_videos(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
    show_all: Annotated[
        bool,
        Query(alias="all", description="Superusers only: everyone's videos"),
    ] = False,
) -> Any:
    """
    Your videos, newest first, each with its rally count and total rally time.
    """
    if show_all and not current_user.is_superuser:
        # An error rather than silently showing your own: a frontend bug shows up
        raise HTTPException(
            status_code=403, detail="The user doesn't have enough privileges"
        )
    videos, count = crud.list_videos(
        session=session,
        owner_id=None if show_all else current_user.id,
        skip=skip,
        limit=limit,
    )
    return VideosPublic(data=videos, count=count)


@router.post("/", response_model=VideoPublic, status_code=201)
def upload_video(
    session: SessionDep, current_user: CurrentUser, file: UploadFile
) -> Any:
    """
    Upload a video (multipart, field `file`). Status starts as `uploaded`.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {ext or '(none)'}; use one of {sorted(VIDEO_EXTENSIONS)}",
        )

    # Name files by id, never by the user's filename (collisions, odd characters, ../ tricks)
    video_id = uuid.uuid4()
    storage_path = f"{video_id}/original{ext}"
    video_dir = settings.STORAGE_DIR / str(video_id)
    video_dir.mkdir(parents=True)
    try:
        with open(settings.STORAGE_DIR / storage_path, "wb") as out:
            shutil.copyfileobj(
                file.file, out, CHUNK
            )  # chunked: never the whole video in memory
        # Reject non-videos now, not later in the worker
        info = media.probe(settings.STORAGE_DIR / storage_path)
    except ValueError as e:
        shutil.rmtree(video_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"{file.filename}: {e}")
    except BaseException:
        shutil.rmtree(video_dir, ignore_errors=True)
        raise

    video = Video(
        id=video_id,
        owner_id=current_user.id,
        filename=(file.filename or f"video{ext}")[:255],
        storage_path=storage_path,
        **info,
    )
    session.add(video)
    session.commit()
    session.refresh(video)
    return crud.get_video_public(session=session, video=video)


@router.get("/{id}", response_model=VideoPublic)
def read_video(session: SessionDep, current_user: CurrentUser, id: uuid.UUID) -> Any:
    """
    Get a video: status (uploaded → queued → processing → done | failed), progress,
    size, court, rally count and total rally time.
    """
    video = get_video_for_user(session, current_user, id)
    return crud.get_video_public(session=session, video=video)


@router.delete(
    "/{id}",
    response_model=Message,
    responses={409: {"description": "The video is being processed"}},
)
def delete_video(session: SessionDep, current_user: CurrentUser, id: uuid.UUID) -> Any:
    """
    Delete a video with its rallies and files (yours, or any as a superuser).
    Not allowed while it's being processed.
    """
    video = get_video_for_user(session, current_user, id)
    video_dir = settings.STORAGE_DIR / str(video.id)
    if not crud.delete_video(session=session, video_id=id):
        session.expire_all()
        if session.get(Video, id) is None:  # deleted by another request meanwhile
            raise HTTPException(status_code=404, detail="Video not found")
        raise HTTPException(
            status_code=409,
            detail="The video is being processed; delete it once processing has finished",
        )
    # Files after the database: if this fails, only orphan files are left, never a
    # video whose files are gone
    shutil.rmtree(video_dir, ignore_errors=True)
    return Message(message="Video deleted successfully")


@router.get(
    "/{id}/frame",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
def read_video_frame(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    t: Annotated[
        float, Query(ge=0, description="Seconds; the very first frame is often black")
    ] = 1.0,
) -> Response:
    """
    One frame as JPEG, for marking the court corners.
    """
    video = get_video_for_user(session, current_user, id)
    t = min(t, max(0.0, (video.duration or 0.0) - 0.5))  # stay inside short videos
    try:
        jpeg = media.frame_jpeg(settings.STORAGE_DIR / video.storage_path, t)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Couldn't extract a frame: {e}")
    return Response(jpeg, media_type="image/jpeg")


@router.put("/{id}/court", response_model=VideoPublic)
def update_video_court(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID, court_in: CourtUpdate
) -> Any:
    """
    Save the 4 court corners (fractions 0-1). Not allowed while queued/processing.
    """
    video = get_video_for_user(session, current_user, id)
    corners = [list(p) for p in court_in.corners]
    if not crud.set_video_court(session=session, video_id=id, corners=corners):
        raise HTTPException(
            status_code=409,
            detail=f"Can't change the court while the video is {video.status}",
        )
    session.refresh(video)
    return crud.get_video_public(session=session, video=video)


@router.post("/{id}/process", response_model=VideoPublic)
def process_video(session: SessionDep, current_user: CurrentUser, id: uuid.UUID) -> Any:
    """
    Queue the video for rally detection.
    """
    video = get_video_for_user(session, current_user, id)
    if not crud.queue_video(session=session, video_id=id):
        session.refresh(video)
        if video.court_corners is None:
            raise HTTPException(
                status_code=409,
                detail="Mark the court corners first (PUT /videos/{id}/court)",
            )
        raise HTTPException(status_code=409, detail=f"Video is already {video.status}")
    session.refresh(video)
    return crud.get_video_public(session=session, video=video)


@router.get("/{id}/rallies", response_model=RalliesPublic)
def read_video_rallies(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID
) -> Any:
    """
    Detected rallies, in seconds, ordered by start time.
    """
    video = get_video_for_user(session, current_user, id)
    return RalliesPublic(data=video.rallies, count=len(video.rallies))  # type: ignore[arg-type]


@router.get("/{id}/file-url", response_model=FileUrl)
def read_video_file_url(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID
) -> Any:
    """
    A temporary link for playing the video in the browser.

    `<video src>` can't send the login header, so use this URL (it carries a
    signed token for this one video) instead of putting the login token in a URL.
    """
    get_video_for_user(session, current_user, id)
    token, expires_at = create_file_token(id)
    return FileUrl(
        url=f"{settings.API_V1_STR}/videos/{id}/file?token={token}",
        expires_at=expires_at,
    )


@router.get("/{id}/file", response_class=FileResponse)
def read_video_file(
    session: SessionDep,
    id: uuid.UUID,
    token: Annotated[str, Query(description="From GET /videos/{id}/file-url")],
) -> FileResponse:
    """
    Stream the video file. Supports Range requests, which the browser's player
    needs for seeking.
    """
    if not verify_file_token(token, id):
        raise HTTPException(status_code=403, detail="Invalid or expired file link")
    video = session.get(Video, id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    path = settings.STORAGE_DIR / video.storage_path
    return FileResponse(
        path,
        media_type=mimetypes.guess_type(path.name)[0] or "video/mp4",
        filename=video.filename,
        content_disposition_type="inline",
    )
