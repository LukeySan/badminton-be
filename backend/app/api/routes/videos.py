import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from sqlmodel import Session

from app import crud, media, thumbnails, trajectory
from app.api.deps import CurrentUser, SessionDep
from app.api.errors import CodedHTTPException
from app.core.config import settings
from app.core.security import create_file_token, verify_file_token
from app.core.upload_limit import format_bytes
from app.models import (
    CourtUpdate,
    ErrorCode,
    ErrorWithCode,
    FileUrl,
    Message,
    RalliesPublic,
    ThumbnailUrls,
    TrajectoryPublic,
    User,
    Video,
    VideoPublic,
    VideosPublic,
)

router = APIRouter(prefix="/videos", tags=["videos"])

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
CHUNK = 1024 * 1024
MAX_TRAJECTORY_SECONDS = 600  # per request: ~18,000 points at 30 fps

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


@router.post(
    "/",
    response_model=VideoPublic,
    status_code=201,
    responses={
        411: {"description": "No Content-Length header"},
        413: {"description": "File larger than MAX_UPLOAD_BYTES (2 GB)"},
        415: {"description": "Not a video file type"},
        422: {
            "description": "Video longer than MAX_VIDEO_MINUTES (60), or invalid form"
        },
    },
)
def upload_video(
    session: SessionDep, current_user: CurrentUser, file: UploadFile
) -> Any:
    """
    Upload a video (multipart, field `file`). Status starts as `uploaded`.
    Limits: MAX_UPLOAD_BYTES (413) and MAX_VIDEO_MINUTES (422).
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
        size = 0
        with open(settings.STORAGE_DIR / storage_path, "wb") as out:
            # Chunked: never the whole video in memory. Counted, because the
            # middleware's Content-Length check includes the multipart wrapping.
            while chunk := file.file.read(CHUNK):
                size += len(chunk)
                if size > settings.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="The file is larger than the "
                        f"{format_bytes(settings.MAX_UPLOAD_BYTES)} limit",
                    )
                out.write(chunk)
        # Reject non-videos now, not later in the worker
        info = media.probe(settings.STORAGE_DIR / storage_path)
        if info["duration"] > settings.MAX_VIDEO_MINUTES * 60:
            raise HTTPException(
                status_code=422,
                detail=f"The video is {info['duration'] / 60:.1f} minutes long; "
                f"the limit is {settings.MAX_VIDEO_MINUTES:g} minutes",
            )
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


@router.post(
    "/{id}/process",
    response_model=VideoPublic,
    responses={
        409: {"description": "No court yet, or already queued/processing"},
        429: {"description": "Too many of your videos queued or processing"},
    },
)
def process_video(session: SessionDep, current_user: CurrentUser, id: uuid.UUID) -> Any:
    """
    Queue the video for rally detection. At most MAX_ACTIVE_VIDEOS_PER_USER of a
    user's videos can be queued or processing at once (429 beyond that).
    """
    video = get_video_for_user(session, current_user, id)
    result = crud.queue_video(
        session=session,
        video_id=id,
        max_active=settings.MAX_ACTIVE_VIDEOS_PER_USER,
    )
    if result == "limit_reached":
        n = settings.MAX_ACTIVE_VIDEOS_PER_USER
        raise HTTPException(
            status_code=429,
            detail=f"You already have {n} video{'s' if n != 1 else ''} queued or "
            "processing; try again when one has finished",
        )
    if result == "not_allowed":
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
    The video's rallies (detected, then corrected by the user), in seconds,
    numbered in time order.
    """
    video = get_video_for_user(session, current_user, id)
    rallies = crud.numbered_rallies(session=session, video_id=video.id)
    return RalliesPublic(data=rallies, count=len(rallies))


@router.get(
    "/{id}/trajectory",
    response_model=TrajectoryPublic,
    responses={
        404: {
            "model": ErrorWithCode,
            "description": "code `trajectory_missing`: the video was analyzed before "
            "trajectories were saved (or not analyzed yet); re-analyzing it adds one. "
            "A video that doesn't exist gives a plain 404 without a code.",
        },
    },
)
def read_video_trajectory(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    start: Annotated[float, Query(ge=0, description="Seconds")],
    end: Annotated[float, Query(ge=0, description="Seconds")],
) -> Any:
    """
    Shuttle positions for every frame from `start` to `end` (seconds), as fractions
    of the frame like the court corners. For drawing the shuttle trail.
    At most 600 seconds per request.
    """
    video = get_video_for_user(session, current_user, id)
    if end <= start:
        raise HTTPException(status_code=422, detail="end must be after start")
    if end - start > MAX_TRAJECTORY_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=f"At most {MAX_TRAJECTORY_SECONDS} seconds per request",
        )
    track = trajectory.load_trajectory(video.id)
    if track is None:
        raise CodedHTTPException(
            404,
            ErrorCode.TRAJECTORY_MISSING,
            "This video has no shuttle trajectory yet; re-analyze it to enable "
            "the shuttle trail",
        )
    return TrajectoryPublic(
        fps=track.fps,
        start=start,
        end=end,
        points=trajectory.points_between(track, start, end),
    )


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


@router.get(
    "/{id}/thumbnails-url",
    response_model=ThumbnailUrls,
    responses={
        404: {
            "model": ErrorWithCode,
            "description": "code `thumbnails_missing`: analyzed before thumbnails "
            "existed, or not analyzed yet (see the backfill command). A video that "
            "doesn't exist gives a plain 404 without a code.",
        }
    },
)
def read_video_thumbnails_url(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID
) -> Any:
    """
    Temporary links (12 h) to the thumbnail strip: one small frame per second,
    packed into sprite sheets. Fetch `index_url` for the layout (interval, tile
    size, columns, rows, count), then the sheets as needed.
    """
    get_video_for_user(session, current_user, id)
    index = thumbnails.read_index(id)
    if index is None:
        raise CodedHTTPException(
            404,
            ErrorCode.THUMBNAILS_MISSING,
            "This video has no thumbnails yet",
        )
    # One token per video covers all its files (video, thumbnails)
    token, expires_at = create_file_token(id)
    base = f"{settings.API_V1_STR}/videos/{id}/thumbnails"
    return ThumbnailUrls(
        index_url=f"{base}/{thumbnails.INDEX}?token={token}",
        sheet_urls=[f"{base}/{name}?token={token}" for name in index["sheets"]],
        expires_at=expires_at,
    )


@router.get(
    "/{id}/thumbnails/{name}",
    response_class=FileResponse,
    responses={
        200: {"content": {"image/jpeg": {}, "application/json": {}}},
        403: {"description": "Invalid or expired link"},
    },
)
def read_video_thumbnail_file(
    id: uuid.UUID,
    name: str,
    token: Annotated[str, Query(description="From GET /videos/{id}/thumbnails-url")],
) -> FileResponse:
    """
    One file of the thumbnail strip: `index.json` or a sheet (`sheet_000.jpg`...).
    """
    if not verify_file_token(token, id):
        raise HTTPException(status_code=403, detail="Invalid or expired file link")
    path = thumbnails.file_path(id, name)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path,
        media_type="application/json" if name == thumbnails.INDEX else "image/jpeg",
        # The link is private (signed) and the files only change on re-analysis
        headers={"Cache-Control": "private, max-age=3600"},
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
