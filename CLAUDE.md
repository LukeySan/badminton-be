# Badminton Rally Analyzer — app repo (beta prototype)

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
| Frontend | **Vite + React + TypeScript** | Single-page app. NOT Next.js (no server inside a Capacitor app). NOT React Native. |
| Routing / data | React Router, TanStack Query | Query handles polling job status |
| Styling | Tailwind | |
| Mobile (later) | Capacitor | Wraps the Vite `dist/` build for iOS/Android |
| API | **FastAPI** (Python 3.11) | |
| DB (beta) | **SQLite** via **SQLAlchemy 2.0 ORM** | One file shared by API and worker. Supabase Postgres later. No raw SQL strings (developer preference); status changes use ORM `update(...).where(status.in_(...))` so check+write is atomic. |
| Storage (beta) | Local folder `storage/` | Supabase Storage later |
| Processing | Python worker: TrackNetV3 (fast mode) + rule-based rally detection | Runs on the Mac (MPS) or the RTX 2070 Super PC (CUDA) |

Dev machine: macOS, Apple Silicon, conda env `badminton`
(`conda activate badminton`; use `python` and `python -m pip`, not `python3`/`pip`).

## Repo layout (target)

```
badminton-app/
  CLAUDE.md
  backend/          FastAPI app (app/main.py, db.py, models, routes)
  worker/           pipeline + job loop
    tracknet/       copied TrackNetV3 inference code (see list below)
    ckpts/          TrackNet_best.pt  (gitignored; download link in TrackNetV3 README)
    rally_detect.py copied from research, rule_detect() is the detector
    pipeline.py     process_video(...)
    run_worker.py   polls DB for queued jobs
  frontend/         Vite React app
  storage/          uploaded videos + per-video outputs (gitignored)
  data/app.db       SQLite (gitignored)
```

## The processing pipeline (worker)

Input: video file + court corners as **fractions 0–1** of width/height
(order: far-left, far-right, near-right, near-left).

1. **Downscale** to 720p with ffmpeg, constant frame rate, no audio for now:
   `ffmpeg -i in.mp4 -vf scale=-2:720 -c:v libx264 -an out_720.mp4`
   (never use `-r` on phone video — it broke timing before).
2. **Shuttle tracking**, TrackNetV3 *fast mode*: `--eval_mode nonoverlap`,
   `--large_video`, **no InpaintNet** (InpaintNet is broken in nonoverlap mode),
   batch size 4 on Mac, 16 on the 2070. Output CSV: `Frame, Visibility, X, Y`
   (pixels of the 720p video). Prefer calling the model from Python rather than
   shelling out to predict.py, but shelling out is fine for the first version.
3. **Convert court fractions → pixels** of the 720p video, write `court.json`
   in the research format: `{"video_size":[w,h],"corners":{"far-left":[x,y],...}}`.
4. **Rally detection**: `rule_detect(csv, video, court, still=0.25, gap=2.5,
   min_length=1.5)` (other params default: min_speed=0.005, still_speed=0.35,
   margin=40). Convert segment frames → seconds with fps.
5. Return `[{start, end, duration}]` in seconds.

TrackNet inference files to copy from `../badminton-test/TrackNetV3/`:
`predict.py`, `model.py`, `dataset.py`, `test.py` (predict imports helpers from it),
`utils/` (`general.py` includes our crash fix in `write_pred_video`), `LICENSE`
(MIT — keep it and credit TrackNetV3). Not needed: train/preprocess/eval scripts.
Extra deps: torch, opencv-python, numpy, pandas, tqdm, parse, pycocotools, pillow.

Expected accuracy (fast mode, tuned rules, match 5, 72 rallies): 60/72 found,
46/72 with start AND end within 2 s, 12 missed, 13 fake.
Speed on the Mac: ~0.5× video length (20-min video ≈ 10 min).

## API (beta)

- `POST /videos` multipart upload → `{id}`; status `uploaded`
- `GET /videos/{id}/frame?t=1.0` → JPEG frame for court tapping
- `PUT /videos/{id}/court` body `{"corners": [[x,y]×4]}` as fractions 0–1
- `POST /videos/{id}/process` → status `queued`
- `GET /videos/{id}` → `{id, filename, status, progress, duration, fps, error, created_at}`
  status: uploaded → queued → processing → done | failed
- `GET /videos/{id}/rallies` → `[{id, start, end, duration}]` (seconds)
- `GET /videos/{id}/file` → stream the video for the web player (supports Range requests)
- `GET /videos` → list
Enable CORS for the Vite dev server (`http://localhost:5173`).

## Data model (SQLite)

- `videos`: id (uuid), filename, storage_path, status, progress, error, duration,
  fps, width, height, court_corners (JSON fractions), created_at, updated_at
- `rallies`: id, video_id, start, end, source ('rules' | 'model'), created_at
- `corrections` (later, but plan for it): rally_id, video_id, original start/end,
  edited start/end, action ('edit'|'delete'|'add'), created_at — future training data
- No users/auth in the beta (single user). Add `user_id` columns later.

Worker job claiming: in one transaction, select the oldest `queued` video and set it
to `processing` so two workers never take the same job (two machines may run workers:
Mac, 2070 PC, 1060 laptop).

## Frontend screens (beta)

1. **Upload** — pick a video file, upload with progress bar
2. **Mark court** — show `/frame`, user taps 4 corners in order, draw the polygon,
   undo/reset, save as fractions
3. **Processing** — poll `GET /videos/{id}` every few seconds; show status/progress
4. **Player** — HTML5 `<video>` + custom scrubber with rallies drawn as chapter
   segments; rally list (sort by time or length, min-length filter); click to jump;
   "skip breaks" mode plays rallies back to back; pad each rally ~1 s on both sides
   when playing (detector boundaries can be ~1–2 s off)
5. Video list / home

Later (mobile): Capacitor, photo-library picker, push notifications, downscale on the
phone before upload (native hardware encoder plugin, not ffmpeg.wasm) and play the
local original file; edit rally boundaries → corrections; highlight export.

## Milestones (do in order, one per session, commit after each)

1. **Worker pipeline CLI**: `python worker/pipeline.py <video> --corners "x,y x,y x,y x,y"`
   prints rallies. Done when results on `../badminton-test/match4_720.mp4` match the
   research scripts.
2. **Backend API** with SQLite + local storage. Done when the whole flow works from
   FastAPI `/docs`.
3. **Worker loop** polling the DB, updating status/progress, saving rallies, handling
   failures (status `failed` + error message). Done when upload → process → rallies works
   end to end.
4. **Frontend: upload + mark court + processing** screens.
5. **Frontend: player** with chapter scrubber and rally list. ← demo milestone
6. Then: tests, README, run script (`make dev` or similar), corrections table.

## Conventions

- Keep the detector behind one function (`detect_rallies(...)`) so the rules can be
  swapped for the ML model (`rally_model.py` in research) later without other changes.
- All times in the API are seconds (float). All court coordinates in the API are
  fractions 0–1; pixels only inside the worker.
- Never commit videos, weights, `storage/`, or the DB. Add a `.gitignore` first.
- Measure detection changes with `score_rallies.py` against labeled matches
  (`../badminton-test/TrackNetV3/results/match*/labels.csv`), not by eye.

## Working style

Britney has a frontend/full-stack background and is learning the ML/backend side.
Explain design choices briefly (she'll need to explain them in interviews), give
copy-pasteable commands for conda on macOS, keep changes small and testable.
