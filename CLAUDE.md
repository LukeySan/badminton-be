# Badminton Rally Analyzer — backend repo (beta prototype)

Read this first. It describes what we're building, the decisions already made, and
the milestones in order. Background research lives in `../badminton-test/`
(see `../CLAUDE.md` for the full research history and numbers).

## What the app does

Players record a match on a phone on a tripod. The app finds every **rally**
(start/end time) in the raw 20–40+ min video and shows them as YouTube-style
**chapters** on the video scrubber, with a rally list, sort/filter by length,
"skip breaks" playback, and (later) clip export.

v1 / beta scope: rally detection only. No shot types, no "my smashes", no
far-side player tracking. Users must record with the **whole court in frame**,
camera behind the baseline, as high as possible.

## Stack (decided)

| Part | Choice | Notes |
|---|---|---|
| Frontend | **Vite + React + TypeScript** | Separate repo, not started yet. NOT Next.js, NOT React Native. |
| Mobile (later) | Capacitor | Wraps the Vite `dist/` build for iOS/Android |
| API | **FastAPI**, based on **full-stack-fastapi-template** (backend part only) | Python 3.14, uv. Keep the template's structure and conventions. |
| Auth | Template's **JWT login** (users, signup, password reset) | Decided: NOT Supabase Auth. |
| DB | **PostgreSQL 18** in Docker (host port **5433**; a local Postgres 15 uses 5432) | **SQLModel** ORM + **Alembic** migrations. No raw SQL strings (developer preference); status changes use ORM `update(...).where(status.in_(...))` so check+write is atomic. |
| Storage (beta) | Local folder `storage/` | Cloud storage later |
| Processing | Python worker: TrackNetV3 (fast mode) + rule-based rally detection | Runs on the Mac (MPS) or the RTX 2070 Super PC (CUDA) |

Dev machine: macOS, Apple Silicon. This repo uses **uv** (`uv sync --all-packages`,
`uv run ...`), NOT the conda env. The conda env `badminton` is only for the research
folder. ffmpeg comes from Homebrew. Docker runs only Postgres, Adminer, Mailpit
(Docker on Mac can't use the GPU, so API/worker run on the host).

## Repos (decided: two separate GitHub repos)

- **This repo = backend** (GitHub: LukeySan/badminton-be): API + worker. They stay
  together because the worker shares the SQLModel models / DB code with the API.
- **Frontend = its own repo**, built later. Not in this folder; don't create `frontend/`
  here. Generate its API client from `http://localhost:8000/api/v1/openapi.json` instead
  of hand-writing types.
- Keep the README here backend-only.

## Repo layout

```
badminton-backend/
  pyproject.toml      uv workspace: members backend, worker (one .venv, one uv.lock)
  compose.yml         db (Postgres), adminer, mailpit
  .env.example        copy to .env (gitignored)
  backend/            template's "app" package
    app/api/routes/   login, users, utils (template) + videos (ours)
    app/models.py     User + Video, Rally, Correction and their API models
    app/crud.py       DB operations incl. set_video_court / queue_video
    app/media.py      ffprobe/ffmpeg helpers
    app/core/         config (settings from .env), db, security (+ file tokens)
    app/alembic/      migrations
    tests/            pytest; uses the app_test database (never the dev one)
    scripts/          prestart.sh, test.sh, lint.sh, format.sh
  worker/             own pyproject (torch, opencv...); package = false
    tracknet/         TrackNetV3 inference code (MIT)
    ckpts/            TrackNet_best.pt (gitignored)
    rally_detect.py   copied from research, rule_detect() is the detector
    detector.py       detect_rallies(): the one function the app calls
    pipeline.py       process_video(...) + CLI
    run_worker.py     (milestone 3) polls DB for queued jobs
  storage/            uploaded videos + per-video outputs (gitignored)
```

Commands (see README for more): `docker compose up -d`; in `backend/`:
`uv run bash scripts/prestart.sh` (migrate + admin user), `uv run fastapi dev`,
`uv run bash scripts/test.sh`, `uv run bash scripts/lint.sh`.
Worker CLI from the repo root: `uv run --package worker python worker/pipeline.py ...`.

## The processing pipeline (worker)

Input: video file + court corners as **fractions 0–1** of width/height
(order: far-left, far-right, near-right, near-left).

1. **Downscale** to 720p with ffmpeg, constant frame rate, no audio for now:
   `ffmpeg -i in.mp4 -vf scale=-2:720 -c:v libx264 -an out_720.mp4`
   (never use `-r` on phone video — it broke timing before). Skipped if already ≤720p.
2. **Shuttle tracking**, TrackNetV3 *fast mode* (`worker/tracknet/infer.py`):
   nonoverlap windows, streamed frames, **no InpaintNet** (broken in nonoverlap mode),
   batch size 4 on Mac, 16 on the 2070. Output CSV: `Frame, Visibility, X, Y`
   (pixels of the 720p video).
3. **Convert court fractions → pixels** of the 720p video, write `court.json`
   in the research format: `{"video_size":[w,h],"corners":{"far-left":[x,y],...}}`.
4. **Rally detection**: `detect_rallies(csv, video, court)` → `rule_detect` with
   still=0.25, gap=2.5, min_length=1.5 (min_speed=0.005, still_speed=0.35, margin=40).
5. Return `[{start, end, duration}]` in seconds.

Verified: on `../badminton-test/match4_720.mp4` the tracking CSV is byte-identical to
the research output and the 31 rallies match `rally_detect.py` exactly.

Expected accuracy (fast mode, tuned rules, match 5, 72 rallies): 60/72 found,
46/72 with start AND end within 2 s, 12 missed, 13 fake.
Speed on the Mac: ~0.5–1× video length.

## API (beta), all under `/api/v1`

Login: `POST /login/access-token` (form: username=email, password) → Bearer token.
All video endpoints need the token, except `/file`. Users see only their own videos
(404 for others'); superusers see all. Lists use the template's `{data, count}` shape.

- `POST /videos/` multipart upload (chunked to disk, probed with ffprobe) → status `uploaded`
- `GET /videos/` list, newest first
- `GET /videos/{id}` → id, owner_id, filename, status, progress, error, duration, fps,
  width, height (displayed, rotation applied), court_corners, created_at, updated_at
  status: uploaded → queued → processing → done | failed (done/failed can be re-queued)
- `GET /videos/{id}/frame?t=1.0` → JPEG for court tapping (t clamped to the video)
- `PUT /videos/{id}/court` `{"corners": [[x,y]×4]}` fractions; validated (4 points, 0–1,
  far above near, left of right); 409 while queued/processing
- `POST /videos/{id}/process` → `queued`; 409 if no court or already queued/processing
- `GET /videos/{id}/rallies` → `{data: [{id, start, end, duration}], count}` (seconds)
- `GET /videos/{id}/file-url` → `{url, expires_at}`: signed 12 h link for one video
- `GET /videos/{id}/file?token=` → the file, Range requests supported. `<video src>`
  can't send headers, hence the token. File tokens have `scope=video-file` and are
  rejected as login tokens (check in `deps.get_current_user`).

CORS: `FRONTEND_HOST` (default `http://localhost:5173`).

## Data model (SQLModel, Postgres)

- `user` (template): id uuid, email, hashed_password, is_active, is_superuser, full_name
- `video`: id uuid, owner_id → user (cascade), filename, storage_path, status (str),
  progress, error, duration, fps, width, height, court_corners (JSONB, none_as_null),
  created_at, updated_at (timestamptz; worker refreshes updated_at as a heartbeat)
- `rally`: id, video_id → video (cascade), start, end, source ('rules' | 'model'), created_at
- `correction` (table exists, unused until milestone 6): rally_id, video_id, original
  start/end, edited start/end, action ('edit'|'delete'|'add'), created_at
- Deleting a user deletes their videos/rallies (DB cascade) AND their files in storage.

Worker job claiming (milestone 3): select the oldest `queued` video with
`with_for_update(skip_locked=True)` and set it to `processing` in the same transaction,
so several workers (Mac, 2070 PC, 1060 laptop) never take the same job. Re-queue
`processing` videos whose `updated_at` heartbeat is stale (worker crashed).

## Frontend screens (beta, in the frontend repo, later)

1. **Upload** — pick a video file, upload with progress bar
2. **Mark court** — show `/frame`, user taps 4 corners in order, draw the polygon,
   undo/reset, save as fractions
3. **Processing** — poll `GET /videos/{id}` every few seconds; show status/progress
4. **Player** — HTML5 `<video>` (src from `/file-url`) + custom scrubber with rallies
   drawn as chapter segments; rally list (sort by time or length, min-length filter);
   click to jump; "skip breaks" mode plays rallies back to back; pad each rally ~1 s on
   both sides when playing (detector boundaries can be ~1–2 s off)
5. Login / signup, video list / home

Later (mobile): Capacitor, photo-library picker, push notifications, downscale on the
phone before upload (native hardware encoder plugin, not ffmpeg.wasm) and play the
local original file; edit rally boundaries → corrections; highlight export.

## Milestones (do in order, one per session, commit after each)

1. ✅ **Worker pipeline CLI** (`worker/pipeline.py`), matches the research scripts.
2. ✅ **Backend API**, then moved onto full-stack-fastapi-template (Postgres, SQLModel,
   Alembic, JWT, pytest, CI).
3. **Worker loop** (`worker/run_worker.py`): claim queued videos, update
   status/progress + heartbeat, save rallies (replace old ones on reprocess), handle
   failures (status `failed` + error message). Done when upload → process → rallies works
   end to end, with tests for claiming/failure.
4. **Frontend: upload + mark court + processing** screens (frontend repo).
5. **Frontend: player** with chapter scrubber and rally list (frontend repo). ← demo milestone
6. Then: corrections endpoints, deployment (API container, CUDA worker), run script.

## Conventions

- Follow the template's patterns: models + API models in `models.py`, DB writes in
  `crud.py`, `SessionDep` / `CurrentUser` dependencies, `{data, count}` lists, Alembic
  migration for every table change (`alembic revision --autogenerate`, then read it).
- `scripts/lint.sh` must pass (mypy strict, ty, ruff) and coverage must stay ≥ 90%
  (CI enforces both).
- Keep the detector behind one function (`detect_rallies(...)`) so the rules can be
  swapped for the ML model (`rally_model.py` in research) later without other changes.
- All times in the API are seconds (float). All court coordinates in the API are
  fractions 0–1; pixels only inside the worker.
- Never commit videos, weights, `storage/`, or `.env`.
- Measure detection changes with `score_rallies.py` against labeled matches
  (`../badminton-test/TrackNetV3/results/match*/labels.csv`), not by eye.
- Git: no `Co-Authored-By: Claude` / "Generated with Claude Code" lines.

## Working style

Britney has a frontend/full-stack background and is learning the ML/backend side.
Explain design choices briefly (she'll need to explain them in interviews), give
copy-pasteable commands for macOS, keep changes small and testable.
