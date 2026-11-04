# Badminton Rally Analyzer: backend

The API and video-processing worker for the Badminton Rally Analyzer. A player uploads
a match video and gets back every **rally** (start/end time), which the app shows as
chapters to jump between. The web/mobile app lives in a separate repo and talks to this
one over HTTP.

How rallies are found: [TrackNetV3](https://github.com/qaz812345/TrackNetV3) locates the
shuttle in every frame, then rule-based detection groups "shuttle moving over your court"
into rallies.

> **Status: beta prototype.** Working: the API and the processing pipeline (CLI).
> Not yet: the background worker loop, so uploaded videos stay `queued`.

## Repo layout

```
backend/    FastAPI API (SQLite via SQLAlchemy, local file storage)
worker/     processing pipeline: downscale → TrackNet shuttle tracking → rally detection
  tracknet/   TrackNetV3 inference code (MIT, see LICENSE there)
  ckpts/      model weights (not in git, see below)
storage/    uploaded videos (created on first run, not in git)
data/       app.db SQLite database (created on first run, not in git)
```

The API and worker share one repo because they share the database models
(`backend/app/models.py`).

## Setup (macOS, once)

You need [conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda is enough).

```bash
conda create -n badminton python=3.11 -y
conda activate badminton
conda install -c conda-forge ffmpeg -y
cd badminton-backend
python -m pip install -r backend/requirements.txt -r worker/requirements.txt
```

**Model weights.** Download `TrackNet_best.pt` from the
[TrackNetV3 README](https://github.com/qaz812345/TrackNetV3) into `worker/ckpts/`, or copy
it from the research folder:

```bash
cp ../badminton-test/TrackNetV3/ckpts/TrackNet_best.pt worker/ckpts/
```

**Check you're using the right Python.** After `conda activate badminton`, this must print
a path containing `envs/badminton`:

```bash
which python
```

If it shows something else (for example a pipenv/virtualenv path), another environment is
still active: run `deactivate`, or open a fresh terminal, and activate again.

## Run the API

From the `badminton-backend/` folder:

```bash
conda activate badminton
uvicorn app.main:app --reload --app-dir backend
```

Open **http://localhost:8000/docs** for interactive docs where you can try every endpoint
in the browser. The database and `storage/` folder are created automatically.

The flow, with curl (or do the same steps in `/docs`):

```bash
# 1. Upload: returns the video, copy its "id"
curl -F file=@match.mp4 localhost:8000/videos
ID=<paste id>

# 2. Get a frame to find the court corners on (opens as a JPEG)
curl "localhost:8000/videos/$ID/frame?t=1" -o frame.jpg

# 3. Save the 4 court corners as fractions 0-1 of width/height,
#    order: far-left, far-right, near-right, near-left
curl -X PUT localhost:8000/videos/$ID/court -H 'content-type: application/json' \
  -d '{"corners": [[0.478,0.347],[0.777,0.369],[0.930,0.943],[0.054,0.781]]}'

# 4. Queue it for processing, then check its status
curl -X POST localhost:8000/videos/$ID/process
curl localhost:8000/videos/$ID

# 5. Rallies (empty until the worker exists)
curl localhost:8000/videos/$ID/rallies
```

| Endpoint | What it does |
|---|---|
| `POST /videos` | Upload a video (multipart, field `file`) |
| `GET /videos` | List videos, newest first |
| `GET /videos/{id}` | Status: `uploaded → queued → processing → done / failed`, plus duration, fps, size, court |
| `GET /videos/{id}/frame?t=1` | JPEG frame for marking the court |
| `PUT /videos/{id}/court` | Court corners (fractions 0-1); not allowed while queued/processing |
| `POST /videos/{id}/process` | Queue for processing (409 if already queued or no court yet) |
| `GET /videos/{id}/rallies` | `[{id, start, end, duration}]` in seconds |
| `GET /videos/{id}/file` | Stream the video (supports seeking) |

## Run the pipeline directly (no API)

Useful for testing detection on one video. Takes about 0.5–1× the video's length on an
Apple Silicon Mac.

```bash
conda activate badminton
python worker/pipeline.py ../badminton-test/match4_720.mp4 \
  --corners "0.478125,0.347222 0.777344,0.369444 0.930469,0.943056 0.053906,0.780556"
```

It prints the rallies and writes intermediate files (720p video, shuttle CSV, court.json,
rallies.json) to `worker/output/<video name>/`. Use `--batch-size 16` on an NVIDIA GPU.

## Configuration

| Env var | Default | |
|---|---|---|
| `BADMINTON_STORAGE` | `storage/` | Where uploaded videos go |
| `BADMINTON_DB` | `data/app.db` | SQLite database file |

## Using it from the frontend

- CORS allows the frontend's Vite dev server (`http://localhost:5173`); edit
  `CORS_ORIGINS` in `backend/app/config.py` for other origins.
- The API schema is at `http://localhost:8000/openapi.json`. Generate TypeScript types
  from it in the frontend repo rather than writing them by hand:
  `npx openapi-typescript http://localhost:8000/openapi.json -o src/api/types.ts`
- All times are seconds (float). All court coordinates are fractions 0-1 of the video's
  displayed width/height (`width`/`height` in `GET /videos/{id}`).

## Recording tips for good results

Camera on a tripod **behind the baseline**, as high as possible, landscape, with the
**whole court in frame** (both baselines). Tracking breaks badly when part of the court
is outside the frame.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'` / `'fastapi'`**: wrong Python, see
  "Check you're using the right Python" above.
- **`ffmpeg not found`**: `conda install -c conda-forge ffmpeg` in the `badminton` env.
- **VS Code shows "Unable to import ..."**: select the interpreter
  (`Cmd+Shift+P` → *Python: Select Interpreter* → `badminton`).

## Why no Docker (yet)

Docker on a Mac runs Linux in a virtual machine that can't use the Apple GPU (MPS), so
shuttle tracking would fall back to the CPU and be several times slower. For one developer
with SQLite and local files, conda is simpler. Docker becomes worth it when deploying: an
API image for the cloud, and a CUDA worker image for the RTX 2070 PC (Docker with NVIDIA's
GPU support works there through WSL2).

## Credits

Shuttle tracking uses [TrackNetV3](https://github.com/qaz812345/TrackNetV3)
(MIT license, `worker/tracknet/LICENSE`).
