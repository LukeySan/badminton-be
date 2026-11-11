# Badminton Rally Analyzer: backend

The API and video-processing worker for the Badminton Rally Analyzer. A player uploads
a match video and gets back every **rally** (start/end time), which the app shows as
chapters to jump between. The web/mobile app lives in a separate repo and talks to this
one over HTTP.

How rallies are found: [TrackNetV3](https://github.com/qaz812345/TrackNetV3) locates the
shuttle in every frame, then rule-based detection groups "shuttle moving over your court"
into rallies.

The API is built on the
[Full Stack FastAPI Template](https://github.com/fastapi/full-stack-fastapi-template):
FastAPI, SQLModel, PostgreSQL, Alembic migrations, JWT login, pytest.

> **Status: beta prototype.** Working end to end: upload → mark court → queue → the
> worker finds the rallies → `GET /videos/{id}/rallies`, rally corrections, thumbnail
> strips, reels (highlight edit lists), shuttle trajectories. Not yet: deployment.

## Repo layout

```
backend/             FastAPI API (the template's "app" package)
  app/
    api/routes/        endpoints: login, users, utils, health, videos, rallies, reels
    api/errors.py      errors with a machine-readable code (trajectory_missing, ...)
    models.py          database tables + API input/output models (SQLModel)
    crud.py            database operations, incl. video status changes
    media.py           ffprobe/ffmpeg helpers (video info, frames)
    trajectory.py      reads the per-video shuttle trajectory file
    thumbnails.py      finds the per-video thumbnail strip files
    reel_ops.py        adjusts reels when rallies change (pure functions)
    core/              settings, database engine, password + token security
    alembic/versions/  database migrations
  tests/             pytest (incl. the worker loop, with a fake pipeline)
  scripts/           prestart (migrate + create admin), test, lint, format
worker/              processing: downscale → TrackNet shuttle tracking → rally detection
  run_worker.py        the job loop: claims queued videos, runs the pipeline
  pipeline.py          one video in, rallies out (also a CLI)
  trajectory_file.py   writes the shuttle trajectory file the API serves
  thumbnail_strip.py   makes the thumbnail strip (sprite sheets + index.json)
  backfill_thumbnails.py  thumbnail strips for videos analyzed before they existed
  tracknet/            TrackNetV3 inference code (MIT, see LICENSE there)
  ckpts/               model weights (not in git, see below)
compose.yml          local Postgres, Adminer (DB browser), Mailpit (test emails)
storage/             per video: original, thumbnails/, trajectory.json.gz, work/ (not in git)
```

The API and worker share one repo (and one uv environment) because the worker uses the
same database models. The worker's heavy packages (PyTorch, OpenCV) live in
`worker/pyproject.toml`, so the API alone doesn't need them.

## Setup (macOS, once)

You need [Homebrew](https://brew.sh), [Docker Desktop](https://www.docker.com/products/docker-desktop/)
and [uv](https://docs.astral.sh/uv/) (`brew install uv`). uv installs Python 3.14 itself.

```bash
brew install ffmpeg
cd badminton-backend
cp .env.example .env          # development defaults; edit if you like
uv sync --all-packages        # API + worker dependencies into .venv/
```

**Model weights** (only needed for processing). Download `TrackNet_best.pt` from the
[TrackNetV3 README](https://github.com/qaz812345/TrackNetV3) into `worker/ckpts/`, or copy
it from the research folder:

```bash
cp ../badminton-test/TrackNetV3/ckpts/TrackNet_best.pt worker/ckpts/
```

## Run

```bash
# 1. Database etc. (Docker Desktop must be running)
docker compose up -d

# 2. Create/upgrade the tables and the admin user (after every pull with new migrations)
cd backend
uv run bash scripts/prestart.sh

# 3. API with auto-reload
uv run fastapi dev

# 4. In a second terminal, from the repo root: the worker that processes videos
uv run --package worker python worker/run_worker.py
```

- **API docs:** http://localhost:8000/docs. Click **Authorize** and log in with
  `FIRST_SUPERUSER` / `FIRST_SUPERUSER_PASSWORD` from `.env` (default
  `admin@example.com` / `changethis`), or create an account with `POST /users/signup`.
- **Database browser (Adminer):** http://localhost:8080. System PostgreSQL, server `db`,
  user `postgres`, password `POSTGRES_PASSWORD` from `.env`, database `app`.
- **Emails (password reset) land in Mailpit:** http://localhost:8025

Postgres listens on **port 5433** on your Mac (not the usual 5432), so it can't clash
with a Postgres installed directly on the machine.

`docker compose down` stops the services and keeps the data; `docker compose down -v`
also deletes the database.

## API

All endpoints are under `/api/v1` and need a login token
(`Authorization: Bearer <token>` from `POST /api/v1/login/access-token`), except
`/videos/{id}/file`, which uses a signed link instead (see below). Users only see their
own videos; the superuser sees all.

| Endpoint | What it does |
|---|---|
| `POST /videos/` | Upload a video (multipart, field `file`). Max 2 GB (413, refused before the upload is read; 411 without `Content-Length`) and 60 minutes (422) |
| `GET /videos/` | Your videos, newest first: `{data, count}` (`count` = all of them, for paging with `skip`/`limit`). Superusers: `?all=true` for everyone's (403 for others) |
| `GET /videos/{id}` | Status `uploaded → queued → processing → done / failed`, progress, duration, fps, displayed width/height, court, `court_out_of_frame` warning, `rally_count`, `play_seconds` (total rally time), `failure_reason` when failed |
| `DELETE /videos/{id}` | Delete a video with its rallies and files (yours, or any as a superuser). 409 while it's being processed |
| `GET /videos/{id}/frame?t=1` | JPEG frame for marking the court |
| `PUT /videos/{id}/court` | 4 court corners as fractions of the frame, order far-left, far-right, near-right, near-left. 0-1 is inside the frame; corners may be off-screen (see below). Not allowed while queued/processing. |
| `POST /videos/{id}/process` | Queue for processing (409 if already queued or no court yet; done/failed videos can be queued again). At most 2 of a user's videos queued or processing at once (429) |
| `GET /videos/{id}/rallies` | `{data: [{id, number, start, end, duration}], count}`, seconds, `number` = 1, 2, 3... in time order |
| `POST /videos/{id}/rallies` | Add a rally `{start, end}` (rally corrections, see below) |
| `PUT /videos/{id}/rallies/{rally_id}` | Set a rally's `{start, end}` |
| `DELETE /videos/{id}/rallies/{rally_id}` | Delete a rally |
| `POST /videos/{id}/rallies/{rally_id}/split` | Split it in two at `{at}` seconds |
| `POST /videos/{id}/rallies/merge` | Merge two neighbouring rallies `{rally_ids: [a, b]}` into a new one |
| `GET /videos/{id}/thumbnails-url` | Signed links (12 h) to the thumbnail strip: `{index_url, sheet_urls, expires_at}`. 404 with `code: "thumbnails_missing"` before it exists |
| `GET /videos/{id}/thumbnails/{name}?token=...` | `index.json` or a sheet (`sheet_000.jpg`) |
| `GET /videos/{id}/trajectory?start=&end=` | Shuttle position for every frame in that time range (max 600 s): `{fps, start, end, points: [{frame, t, visible, x, y}]}`, x/y as fractions like the court. 404 with `code: "trajectory_missing"` for videos analyzed before trajectories were saved. Still saved for every video; nothing in the app depends on it now |
| `GET /videos/{id}/reels` | The video's reels, most recently saved first: `{data, count}` |
| `POST /videos/{id}/reels` | Create a reel `{title, edit}` (see Reels below) |
| `GET /reels/{id}` | One reel with its edit list |
| `PUT /reels/{id}` | Save: replaces title and edit list. Send the `updated_at` you loaded; 409 with `code: "reel_changed"` if it was saved elsewhere since |
| `DELETE /reels/{id}` | Soft delete (hidden, row kept) |
| `GET /videos/{id}/file-url` | A temporary link (12 h) for playing the video |
| `GET /videos/{id}/file?token=...` | The video file; supports Range requests (seeking) |

`GET /health` (no login): database reachable, number of queued and processing videos,
and how long the oldest queued one has waited; 503 if the database is down. Queued
videos piling up with nothing processing means no worker is running.

Plus the template's endpoints: login, password recovery, signup, `/users/me`, user
admin, health check.

**Error codes.** Errors are `{"detail": "..."}` as usual. The few the app reacts to with
something other than showing the message also carry a `code`
(`{"detail": "...", "code": "trajectory_missing"}`); match on the code, never the text.

**Reels** are non-destructive highlight edits: the original video is never changed; a
reel is an edit list the app plays back. All times are seconds in the original video.

```jsonc
{
  "version": 2,
  "view": "cut",            // "cut" (rallies only) or "full"
  "format": "16:9",         // or "9:16"
  "lead_in": 1.0,           // pieces may start up to this long before their rally (0-3 s)
  "lead_out": 1.0,          // ...and end this long after it (0-3 s)
  "clips": [                // in reel order, one rally each
    {"rally_id": 271, "pieces": [               // sorted, not overlapping (may touch)
      {"start": 1.5, "end": 3.0, "slow_mo": false},
      {"start": 3.0, "end": 5.2, "slow_mo": true}   // a split: two touching pieces
    ]}
  ],
  "restored_gaps": [{"start": 4.43, "end": 6.0}],   // breaks shown again in cut view
  "overlays": {"trail": true, "captions": [{"text": "What a smash", "start": 3, "end": 4}]}
}
```

Splitting a piece makes two pieces; deleting a piece removes it; a clip with no pieces
left is dropped when the reel is saved. Rejected with 422 (with the usual `loc`, e.g.
`["body","edit","clips",0,"pieces",1,"end"]`): a `rally_id` that isn't one of the
video's rallies, a piece outside its rally's bounds widened by `lead_in`/`lead_out`,
times past the end of the video (0.1 s tolerance), unsorted or overlapping pieces,
captions over 80 characters, unknown fields, `version` other than 2.

v1 (one `trim_start`/`trim_end` per clip) was converted by a migration: each trim became
one piece, with `lead_in`/`lead_out` raised (up to 3 s) so the old trims still fit.

**Rally corrections** (add, edit, delete, split, merge) each return what changed:

```jsonc
{
  "added": [{"id": 3477, "number": 2, "start": 6.0, "end": 8.0, "duration": 2.0}],
  "updated": [{"id": 3474, "number": 1, "start": 2.0, "end": 6.0, "duration": 4.0}],
  "deleted": [],
  "rallies": [ /* the whole list, renumbered in time order */ ],
  "reels": [{"id": "...", "title": "Best rallies", "updated_at": "...",
             "clips_removed": 0, "pieces_trimmed": 1, "pieces_removed": 0}]
}
```

- Rallies can't overlap (touching is fine): 409 otherwise. Merge only takes neighbours.
  Not allowed while the video is queued or processing (409): re-analysis replaces all
  rallies.
- Reels follow: pieces outside a rally's new bounds (+ lead) are trimmed; a deleted
  rally's clips are dropped; a split splits the clip the same way; a merge points both
  rallies' clips at the new rally and joins a clip of the first directly followed by one
  of the second. Changed reels get a new `updated_at`, so an editor with an old copy
  gets 409 `reel_changed` and reloads.
- Every change is logged in the `correction` table with the detector's original bounds
  (kept on each rally as `detected_start`/`detected_end`), the new bounds and who made
  it: detector output vs. the user's correction, for training a better detector later.
- Re-analyzing a video replaces all its rallies (corrections included) with the
  detector's; reel clips move to the new rally overlapping them most.

**Thumbnail strip.** One 160 px wide frame per second, in sprite sheets of 10 x 10
tiles. Fetch `index_url` first:

```json
{"version": 1, "interval": 1.0, "count": 425, "tile_width": 160, "tile_height": 90,
 "columns": 10, "rows": 10, "sheets": ["sheet_000.jpg", "sheet_001.jpg", "..."]}
```

The frame at `t` seconds is tile `i = floor(t / interval)` (up to `count - 1`): in sheet
`i // (columns * rows)` (its URL is `sheet_urls[...]`), at column `i % columns`, row
`(i // columns) % rows`. Several sheets rather than one tall image: an hour would be one
1600 x 32,400 image, more than phones decode reliably, and the app can load only the
sheets near the playhead.

**Why `file-url`:** a browser `<video src="...">` can't send the `Authorization` header.
Instead of putting the 8-day login token in a URL (where it ends up in logs and
history), the frontend asks for a short-lived link that only works for that one video.

Example with curl (uses `jq`: `brew install jq`):

```bash
B=http://localhost:8000/api/v1
TOKEN=$(curl -s -X POST $B/login/access-token \
  -d "username=admin@example.com&password=changethis" | jq -r .access_token)
H="Authorization: Bearer $TOKEN"

curl -H "$H" -F file=@match.mp4 $B/videos/                  # copy the "id"
ID=<paste id>
curl -H "$H" "$B/videos/$ID/frame?t=1" -o frame.jpg
curl -H "$H" -X PUT $B/videos/$ID/court -H 'content-type: application/json' \
  -d '{"corners": [[0.478,0.347],[0.777,0.369],[0.930,0.943],[0.054,0.781]]}'
curl -H "$H" -X POST $B/videos/$ID/process
curl -H "$H" $B/videos/$ID/rallies
```

## Tests and checks

```bash
cd backend
uv run bash scripts/test.sh     # pytest + coverage report (htmlcov/index.html)
uv run bash scripts/lint.sh     # mypy (strict), ty, ruff
uv run bash scripts/format.sh   # auto-format
```

Tests use a separate database, `app_test` (created by `scripts/initdb/` when the
Postgres volume is first made), so they never touch your development data. Uploaded test
videos go to a temporary folder. GitHub Actions runs lint + tests on every push to `main`
and on pull requests, and fails below 90% coverage.

The worker's own failure checks (unreadable video, court not visible) need OpenCV, which
CI doesn't install; run them locally from the repo root:

```bash
uv run pytest worker/tests -q
```

## Database changes (migrations)

After changing a table in `backend/app/models.py`:

```bash
cd backend
uv run alembic revision --autogenerate -m "Add xyz to video"
# read the generated file in app/alembic/versions/, then:
uv run alembic upgrade head
```

## The worker

```bash
uv run --package worker python worker/run_worker.py                  # until Ctrl+C
uv run --package worker python worker/run_worker.py --once           # wait for one job, then exit
uv run --package worker python worker/run_worker.py --batch-size 16  # on an NVIDIA GPU
```

It checks the queue every 5 s (`--poll`), takes the oldest queued video, and sets it to
`processing` → `done` (rallies saved) or `failed` (with the error in `error`).
Reprocessing a video replaces its old rallies, in the same transaction that marks it
done, so the app never sees half-saved results. A failed reprocess keeps the old ones.

You can run several workers at once, on one machine or on several machines using the
same database:

- **Each job goes to exactly one worker.** Claiming uses `SELECT ... FOR UPDATE SKIP
  LOCKED`: the row is locked while one worker claims it, and the others skip it and take
  the next video instead of waiting.
- **Heartbeat.** While a job runs, a background thread updates the video's `progress`
  and `updated_at` every 5 s, using the database's clock (not each machine's, which may
  differ).
- **Crashed workers.** Before claiming, every worker re-queues videos whose heartbeat is
  older than 2 minutes (`--stale-after`). After 3 attempts the video is marked `failed`
  instead, so a video that crashes the worker every time doesn't loop forever.
- **Claim token.** Each claim gets a new `claim_id`, and a worker only writes while the
  video's `claim_id` is still its own. A worker that was wrongly declared dead (laptop
  asleep) stops when it wakes up instead of overwriting the other worker's results.
- **Ctrl+C** puts the current video straight back in the queue.

Intermediate files go to `storage/<video id>/work/`. Two outputs for the app are moved
to `storage/<video id>/` just before the video is marked done, so "done" always comes
with them (a failed run keeps the old ones):

- `thumbnails/`: the thumbnail strip, made from the 720p analysis copy right after the
  downscale (~3 s for 7 minutes; ~320 KB per sheet of 100 tiles).
- `trajectory.json.gz`: the shuttle position per frame (visible + x/y as fractions,
  gzipped JSON, ~50 KB for 21 minutes).

Videos analyzed before thumbnails existed get them from the backfill command (needs
ffmpeg, not torch; safe to run while workers run, and to run again):

```bash
uv run --package worker python worker/backfill_thumbnails.py --dry-run   # what's missing
uv run --package worker python worker/backfill_thumbnails.py             # make them
uv run --package worker python worker/backfill_thumbnails.py --video <id> --force
```

**Log lines.** One line per phase, prefixed with the video's short id so jobs from
several workers can be told apart:

```
[588c9f87] claimed: clipped_vid.mp4, 1.1 min, attempt 1/3
[588c9f87] prepared: 1280x720, 30.00 fps, 1.1 min (downscale 6s, thumbnails 1s)
[588c9f87] tracking 10%        ... one line per 10%
[588c9f87] detecting rallies (tracking took 41s)
[588c9f87] done: 6 rallies (0.3 min of play) in 48s
```

or `failed: <reason> (<technical details>)`, `lost: ...` (another worker took over) or
`stopped (Ctrl+C)`. Problems with the video are logged in one line; unexpected errors
(bugs) also log their traceback.

**Failure reasons.** A failed video gets a short `failure_reason` for the user, and the
technical details in `error` (database and logs only, never in the API: it can contain
server paths):

| `failure_reason` | When |
|---|---|
| Couldn't read this video | ffmpeg/OpenCV can't decode it |
| Court not visible | The shuttle is inside the marked court in under 5% of frames (real matches: 25-94%; wrong corners: 0-6%) |
| Processing kept stopping on this video | The worker stopped responding 3 times on it |
| Something went wrong while processing this video | Anything else (a bug: see the worker log) |

## Run the pipeline directly (no API)

Useful for testing detection on one video. Takes about half the video's length on an
Apple Silicon Mac (7-min match: 3.7 min; 21-min match: 10.6 min).

```bash
uv run --package worker python worker/pipeline.py ../badminton-test/match4_720.mp4 \
  --corners "0.478125,0.347222 0.777344,0.369444 0.930469,0.943056 0.053906,0.780556"
```

It prints the rallies and writes intermediate files (720p video, shuttle CSV, court.json,
rallies.json) to `worker/output/<video name>/`. Use `--batch-size 16` on an NVIDIA GPU.

**Where the time goes** (match 4, 7.1 min of 720p video, M-series Mac):

| Phase | Time |
|---|---|
| Count frames | 3 s |
| Background (median) image | 10 s |
| Model load | < 1 s |
| Shuttle tracking | ~210 s |
| Rally detection | < 0.1 s |

The background image is the median of 300 frames, read front to back
(`MEDIAN_SAMPLES` in `worker/tracknet/infer.py`). TrackNet's own version seeks to
~1800 frames and took 124 s and 15 GB of memory for the same video, with no
accuracy benefit: same score on match 4, and 72 of 73 rallies unchanged on match 5.
Below ~150 frames, accuracy started to drop.

## Configuration

Settings come from `.env` (see `.env.example` for all of them). The main ones:

| Variable | Default | |
|---|---|---|
| `SECRET_KEY` | `changethis` | Signs login tokens and file links. **Change for anything public.** |
| `FIRST_SUPERUSER` / `_PASSWORD` | `admin@example.com` / `changethis` | Admin account created by `prestart.sh` |
| `DATABASE_URL` | Docker Postgres on port 5433, database `app` | |
| `TEST_DATABASE_URL` | same server, database `app_test` | Used only by tests |
| `STORAGE_DIR` | `storage/` | Uploaded videos |
| `FRONTEND_HOST` | `http://localhost:5173` | Allowed by CORS (the Vite dev server) |
| `FILE_TOKEN_EXPIRE_MINUTES` | 720 | How long `file-url` links work |
| `MAX_UPLOAD_BYTES` | 2000000000 | Largest upload (2 GB, as phones count it) |
| `MAX_VIDEO_MINUTES` | 60 | Longest video |
| `MAX_ACTIVE_VIDEOS_PER_USER` | 2 | Videos one user can have queued or processing at once |

`.env` is gitignored: never commit real secrets. With `FASTAPI_ENV=development` the
`changethis` defaults only cause warnings; without it the app refuses to start with them.

## Using it from the frontend

- The OpenAPI schema is at `http://localhost:8000/api/v1/openapi.json`. Generate the
  TypeScript client from it in the frontend repo rather than writing types by hand.
- All times are seconds (float). Court coordinates are fractions of the video's
  displayed `width`/`height`: 0-1 is inside the frame.
- **Off-screen court corners.** A phone on a tripod often cuts off a near corner, so
  the court-tapping screen should show a gray margin around the frame (like the
  research tool `pick_court.py`) where those corners can be tapped. A tap in the margin
  is just a fraction below 0 or above 1. The backend decides what that means; the app
  only shows the result:

  | Furthest corner outside the frame | API |
  |---|---|
  | up to 10% (-0.1 to 1.1) | accepted, `court_out_of_frame: false` |
  | up to 30% (-0.3 to 1.3) | accepted, `court_out_of_frame: true`: tell the user "part of your court is out of frame, results may be less accurate" and let them continue or re-mark |
  | further | 422: a mis-tap, not a court |

  `court_out_of_frame` is on every video response, so any screen can show it. The
  limits are `COURT_MARGIN_OK` / `COURT_MARGIN_MAX` in `backend/app/models.py`.
- Play a video with `<video src={API_HOST + fileUrl.url}>` from `GET /videos/{id}/file-url`.
  Fetch the court frame with the login header (e.g. into a blob URL).

## Recording tips for good results

Camera on a tripod **behind the baseline**, as high as possible, landscape, with the
**whole court in frame** (both baselines). Tracking breaks badly when part of the court
is outside the frame.

## Troubleshooting

- **`ffprobe not found` / `ffmpeg not found`**: `brew install ffmpeg`.
- **`Cannot connect to the Docker daemon`**: start Docker Desktop.
- **`connection refused` on port 5433**: `docker compose up -d`.
- **`relation "video" does not exist`** or **`column video.claim_id does not exist`**:
  run `uv run bash scripts/prestart.sh` in `backend/` (applies new migrations).
- **Videos stay `queued`**: the worker isn't running (`worker/run_worker.py`).
- **A video failed with `FileNotFoundError ... TrackNet_best.pt`**: copy the model weights
  into `worker/ckpts/` (see Setup), then queue the video again.
- **A different Python environment is active** (uv warns that `VIRTUAL_ENV` doesn't
  match): harmless, uv uses the project's `.venv`. Run `deactivate` to silence it.
- **VS Code shows "Unable to import ..."**: *Python: Select Interpreter* →
  `badminton-backend/.venv`.

## Why Docker only for the database

Docker on a Mac runs Linux in a virtual machine that can't use the Apple GPU, so shuttle
tracking inside a container would run on the CPU, several times slower. The API and
worker therefore run directly on the machine with uv; only Postgres, Adminer and Mailpit
run in Docker. Containers for the API and a CUDA worker come with deployment.

## Credits

- [Full Stack FastAPI Template](https://github.com/fastapi/full-stack-fastapi-template)
  by Sebastián Ramírez, MIT license (`backend/TEMPLATE_LICENSE`)
- [TrackNetV3](https://github.com/qaz812345/TrackNetV3), MIT license
  (`worker/tracknet/LICENSE`)
