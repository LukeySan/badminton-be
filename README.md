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
> worker finds the rallies → `GET /videos/{id}/rallies`. Not yet: rally corrections,
> deployment.

## Repo layout

```
backend/             FastAPI API (the template's "app" package)
  app/
    api/routes/        endpoints: login, users, utils, videos
    models.py          database tables + API input/output models (SQLModel)
    crud.py            database operations, incl. video status changes
    media.py           ffprobe/ffmpeg helpers (video info, frames)
    core/              settings, database engine, password + token security
    alembic/versions/  database migrations
  tests/             pytest (incl. the worker loop, with a fake pipeline)
  scripts/           prestart (migrate + create admin), test, lint, format
worker/              processing: downscale → TrackNet shuttle tracking → rally detection
  run_worker.py        the job loop: claims queued videos, runs the pipeline
  pipeline.py          one video in, rallies out (also a CLI)
  tracknet/            TrackNetV3 inference code (MIT, see LICENSE there)
  ckpts/               model weights (not in git, see below)
compose.yml          local Postgres, Adminer (DB browser), Mailpit (test emails)
storage/             uploaded videos (created on first upload, not in git)
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
| `POST /videos/` | Upload a video (multipart, field `file`) |
| `GET /videos/` | Your videos, newest first: `{data, count}` |
| `GET /videos/{id}` | Status `uploaded → queued → processing → done / failed`, progress, duration, fps, displayed width/height, court |
| `GET /videos/{id}/frame?t=1` | JPEG frame for marking the court |
| `PUT /videos/{id}/court` | 4 court corners as fractions 0-1, order far-left, far-right, near-right, near-left. Not allowed while queued/processing. |
| `POST /videos/{id}/process` | Queue for processing (409 if already queued or no court yet; done/failed videos can be queued again) |
| `GET /videos/{id}/rallies` | `{data: [{id, start, end, duration}], count}`, seconds |
| `GET /videos/{id}/file-url` | A temporary link (12 h) for playing the video |
| `GET /videos/{id}/file?token=...` | The video file; supports Range requests (seeking) |

Plus the template's endpoints: login, password recovery, signup, `/users/me`, user
admin, health check.

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

Intermediate files go to `storage/<video id>/work/`.

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

`.env` is gitignored: never commit real secrets. With `FASTAPI_ENV=development` the
`changethis` defaults only cause warnings; without it the app refuses to start with them.

## Using it from the frontend

- The OpenAPI schema is at `http://localhost:8000/api/v1/openapi.json`. Generate the
  TypeScript client from it in the frontend repo rather than writing types by hand.
- All times are seconds (float). Court coordinates are fractions 0-1 of the video's
  displayed `width`/`height`.
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
