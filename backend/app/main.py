"""
Rally Analyzer API (beta: SQLite + local storage, single user).

Run from the repo root:
    uvicorn app.main:app --reload --app-dir backend
then open http://localhost:8000/docs
"""

import mimetypes
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from . import db, media
from .config import CORS_ORIGINS, STORAGE_DIR, VIDEO_EXTENSIONS
from .schemas import Court, RallyOut, VideoOut

CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(app):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    conn = db.connect()
    db.init(conn)
    conn.close()
    yield


app = FastAPI(title="Badminton Rally Analyzer", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


def get_conn():
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


def video_or_404(conn, video_id):
    row = db.get_video(conn, video_id)
    if row is None:
        raise HTTPException(404, "video not found")
    return row


# Handlers are plain `def` (not async): FastAPI runs them in a thread pool, so slow
# file copies and ffmpeg calls don't block other requests.

@app.post("/videos", response_model=VideoOut, status_code=201)
def upload_video(file: UploadFile = File(...), conn=Depends(get_conn)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(415, f"unsupported file type {ext or '(none)'}; use one of {sorted(VIDEO_EXTENSIONS)}")

    # Name files by id, never by the user's filename (collisions, odd characters, ../ tricks).
    video_id = db.new_id()
    video_dir = STORAGE_DIR / video_id
    video_dir.mkdir(parents=True)
    rel_path = f"{video_id}/original{ext}"
    try:
        with open(STORAGE_DIR / rel_path, "wb") as out:
            shutil.copyfileobj(file.file, out, CHUNK)  # chunked: never the whole video in memory
        info = media.probe(STORAGE_DIR / rel_path)  # reject non-videos now, not later in the worker
    except ValueError as e:
        shutil.rmtree(video_dir, ignore_errors=True)
        raise HTTPException(400, f"{file.filename}: {e}")
    except BaseException:
        shutil.rmtree(video_dir, ignore_errors=True)
        raise

    db.insert_video(conn, video_id, file.filename, rel_path, info)
    return db.video_to_dict(db.get_video(conn, video_id))


@app.get("/videos", response_model=list[VideoOut])
def list_videos(conn=Depends(get_conn)):
    return [db.video_to_dict(r) for r in db.list_videos(conn)]


@app.get("/videos/{video_id}", response_model=VideoOut)
def get_video(video_id: str, conn=Depends(get_conn)):
    return db.video_to_dict(video_or_404(conn, video_id))


@app.get("/videos/{video_id}/frame", response_class=Response,
         responses={200: {"content": {"image/jpeg": {}}}})
def get_frame(video_id: str, t: float = Query(1.0, ge=0, description="seconds; the first frame is often black"),
              conn=Depends(get_conn)):
    row = video_or_404(conn, video_id)
    t = min(t, max(0.0, row["duration"] - 0.5))  # stay inside short videos
    try:
        jpeg = media.frame_jpeg(STORAGE_DIR / row["storage_path"], t)
    except RuntimeError as e:
        raise HTTPException(500, f"couldn't extract a frame: {e}")
    return Response(jpeg, media_type="image/jpeg")


@app.put("/videos/{video_id}/court", response_model=VideoOut)
def set_court(video_id: str, court: Court, conn=Depends(get_conn)):
    row = video_or_404(conn, video_id)
    if not db.set_court(conn, video_id, [list(p) for p in court.corners]):
        raise HTTPException(409, f"can't change the court while the video is {row['status']}")
    return db.video_to_dict(db.get_video(conn, video_id))


@app.post("/videos/{video_id}/process", response_model=VideoOut)
def process_video(video_id: str, conn=Depends(get_conn)):
    if not db.queue(conn, video_id):
        row = video_or_404(conn, video_id)
        if row["court_corners"] is None:
            raise HTTPException(409, "mark the court corners first (PUT /videos/{id}/court)")
        raise HTTPException(409, f"video is already {row['status']}")
    return db.video_to_dict(db.get_video(conn, video_id))


@app.get("/videos/{video_id}/rallies", response_model=list[RallyOut])
def get_rallies(video_id: str, conn=Depends(get_conn)):
    video_or_404(conn, video_id)
    return [{**dict(r), "duration": round(r["end"] - r["start"], 2)} for r in db.get_rallies(conn, video_id)]


@app.get("/videos/{video_id}/file")
def get_file(video_id: str, conn=Depends(get_conn)):
    # FileResponse answers Range requests with 206 Partial Content, which the
    # browser's <video> needs to seek (Safari won't play without it).
    row = video_or_404(conn, video_id)
    path = STORAGE_DIR / row["storage_path"]
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "video/mp4",
                        filename=row["filename"], content_disposition_type="inline")
