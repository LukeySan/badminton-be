"""
Rally Analyzer API (beta: SQLite + local storage, single user).

Run from the repo root:
    uvicorn app.main:app --reload --app-dir backend
then open http://localhost:8000/docs
"""

import mimetypes
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import db, media
from .config import CORS_ORIGINS, STORAGE_DIR, VIDEO_EXTENSIONS
from .models import Video
from .schemas import Court, RallyOut, VideoOut

CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(app):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    db.init()
    yield


app = FastAPI(title="Badminton Rally Analyzer", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


def video_or_404(session, video_id) -> Video:
    video = session.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    return video


# Handlers are plain `def` (not async): FastAPI runs them in a thread pool, so slow
# file copies and ffmpeg calls don't block other requests.

@app.post("/videos", response_model=VideoOut, status_code=201)
def upload_video(file: UploadFile = File(...), session: Session = Depends(db.get_session)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(415, f"unsupported file type {ext or '(none)'}; use one of {sorted(VIDEO_EXTENSIONS)}")

    # Name files by id, never by the user's filename (collisions, odd characters, ../ tricks).
    video_id = str(uuid.uuid4())  # needed before the insert: it names the storage folder
    video = Video(id=video_id, filename=file.filename, storage_path=f"{video_id}/original{ext}")
    video_dir = STORAGE_DIR / video_id
    video_dir.mkdir(parents=True)
    try:
        with open(STORAGE_DIR / video.storage_path, "wb") as out:
            shutil.copyfileobj(file.file, out, CHUNK)  # chunked: never the whole video in memory
        info = media.probe(STORAGE_DIR / video.storage_path)  # reject non-videos now, not later in the worker
    except ValueError as e:
        shutil.rmtree(video_dir, ignore_errors=True)
        raise HTTPException(400, f"{file.filename}: {e}")
    except BaseException:
        shutil.rmtree(video_dir, ignore_errors=True)
        raise

    video.duration, video.fps = info["duration"], info["fps"]
    video.width, video.height = info["width"], info["height"]
    session.add(video)
    session.commit()
    return video


@app.get("/videos", response_model=list[VideoOut])
def list_videos(session: Session = Depends(db.get_session)):
    return session.scalars(select(Video).order_by(Video.created_at.desc())).all()


@app.get("/videos/{video_id}", response_model=VideoOut)
def get_video(video_id: str, session: Session = Depends(db.get_session)):
    return video_or_404(session, video_id)


@app.get("/videos/{video_id}/frame", response_class=Response,
         responses={200: {"content": {"image/jpeg": {}}}})
def get_frame(video_id: str, t: float = Query(1.0, ge=0, description="seconds; the first frame is often black"),
              session: Session = Depends(db.get_session)):
    video = video_or_404(session, video_id)
    t = min(t, max(0.0, video.duration - 0.5))  # stay inside short videos
    try:
        jpeg = media.frame_jpeg(STORAGE_DIR / video.storage_path, t)
    except RuntimeError as e:
        raise HTTPException(500, f"couldn't extract a frame: {e}")
    return Response(jpeg, media_type="image/jpeg")


@app.put("/videos/{video_id}/court", response_model=VideoOut)
def set_court(video_id: str, court: Court, session: Session = Depends(db.get_session)):
    video = video_or_404(session, video_id)
    if not db.set_court(session, video_id, [list(p) for p in court.corners]):
        raise HTTPException(409, f"can't change the court while the video is {video.status}")
    return video  # the commit expired it, so this reloads the new values


@app.post("/videos/{video_id}/process", response_model=VideoOut)
def process_video(video_id: str, session: Session = Depends(db.get_session)):
    if not db.queue(session, video_id):
        video = video_or_404(session, video_id)
        if video.court_corners is None:
            raise HTTPException(409, "mark the court corners first (PUT /videos/{id}/court)")
        raise HTTPException(409, f"video is already {video.status}")
    return session.get(Video, video_id)


@app.get("/videos/{video_id}/rallies", response_model=list[RallyOut])
def get_rallies(video_id: str, session: Session = Depends(db.get_session)):
    return video_or_404(session, video_id).rallies  # ordered by start (see models.Video)


@app.get("/videos/{video_id}/file")
def get_file(video_id: str, session: Session = Depends(db.get_session)):
    # FileResponse answers Range requests with 206 Partial Content, which the
    # browser's <video> needs to seek (Safari won't play without it).
    video = video_or_404(session, video_id)
    path = STORAGE_DIR / video.storage_path
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "video/mp4",
                        filename=video.filename, content_disposition_type="inline")
