"""
Video in, rallies out.

    python worker/pipeline.py video.mp4 --corners "0.48,0.35 0.78,0.37 0.93,0.94 0.05,0.78"

Corners are fractions 0-1 of the frame width/height, in the order
far-left, far-right, near-right, near-left (the API's format). Pixels exist only
inside this module.

Steps: downscale to 720p -> thumbnail strip (sprite sheets for the app's scrubber)
-> TrackNet shuttle tracking (fast mode) -> trajectory file
(shuttle positions as fractions, for the app's shuttle trail) -> court.json
-> rule-based rally detection. Intermediate files go to --work-dir.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import detect_rallies
from job_errors import VideoUnreadable
from tracknet.infer import track_shuttle
from thumbnail_strip import DIRNAME as THUMBNAILS_DIRNAME
from thumbnail_strip import generate_thumbnails
from trajectory_file import FILENAME as TRAJECTORY_FILENAME
from trajectory_file import write_trajectory

WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(WORKER_DIR, "ckpts", "TrackNet_best.pt")
CORNER_NAMES = ["far-left", "far-right", "near-right", "near-left"]
TARGET_HEIGHT = 720


def video_info(path):
    """Size, fps, frames. Raises VideoUnreadable unless OpenCV can decode a frame (the
    API only checked the file with ffprobe, which accepts more than OpenCV reads)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened() or not cap.read()[0]:
        cap.release()
        raise VideoUnreadable(f"OpenCV can't open or decode {os.path.basename(path)}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration"] = info["frames"] / info["fps"] if info["fps"] else 0.0
    return info


def downscale(src, dst):
    """Re-encode to 720p. No -r: forcing a frame rate on phone video broke timestamps before."""
    if video_info(src)["height"] <= TARGET_HEIGHT:
        return src  # already small enough; re-encoding would only lose quality
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-vf", f"scale=-2:{TARGET_HEIGHT}", "-c:v", "libx264", "-an", dst],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise VideoUnreadable(f"ffmpeg downscale failed: {result.stderr.strip()[-500:]}")
    return dst


def write_court_json(corners, width, height, path):
    """Fractions -> pixels of the analysed video, in the research court.json format."""
    px = {name: [round(x * width), round(y * height)] for name, (x, y) in zip(CORNER_NAMES, corners)}
    with open(path, "w") as fh:
        json.dump({"video_size": [width, height], "corners": px}, fh, indent=2)
    return path


def process_video(video_path, corners, work_dir, ckpt_path=DEFAULT_CKPT, batch_size=4,
                  on_progress=None, log=print):
    """Find rallies in video_path. corners: 4 (x, y) fractions. Returns [{start, end, duration}] in seconds.
    on_progress(fraction 0-1) is called during tracking, the slow step."""
    # Same limits as the API (app.models.COURT_MARGIN_MAX): corners may be off-screen
    # by up to 30% of the frame; pixel coordinates are then negative or past the edge.
    if len(corners) != 4 or not all(-0.3 <= v <= 1.3 for c in corners for v in c):
        raise ValueError("corners must be 4 (x, y) pairs with values between -0.3 and 1.3")
    os.makedirs(work_dir, exist_ok=True)

    # One log line per phase; the caller logs tracking progress via on_progress
    t0 = time.time()
    video = downscale(video_path, os.path.join(work_dir, "video_720.mp4"))
    info = video_info(video)
    t1 = time.time()
    # From the analysis copy: same frames and timing as the original, faster to decode
    generate_thumbnails(video, os.path.join(work_dir, THUMBNAILS_DIRNAME))
    log(f"prepared: {info['width']}x{info['height']}, {info['fps']:.2f} fps, "
        f"{info['duration'] / 60:.1f} min (downscale {t1 - t0:.0f}s, "
        f"thumbnails {time.time() - t1:.0f}s)")

    t0 = time.time()
    csv_path = os.path.join(work_dir, "ball.csv")
    track_shuttle(video, ckpt_path, csv_path, batch_size=batch_size,
                  on_progress=on_progress and (lambda done, total: on_progress(done / max(total, 1))))
    log(f"detecting rallies (tracking took {time.time() - t0:.0f}s)")
    # Same frames and timing as the original (no -r when downscaling), so frame / fps
    # is a time in the original video, and fractions don't depend on the resolution
    write_trajectory(csv_path, info["width"], info["height"], info["fps"],
                     os.path.join(work_dir, TRAJECTORY_FILENAME))

    court_path = write_court_json(corners, info["width"], info["height"], os.path.join(work_dir, "court.json"))
    rallies = detect_rallies(csv_path, video, court_path)
    with open(os.path.join(work_dir, "rallies.json"), "w") as fh:
        json.dump({"fps": info["fps"], "rallies": rallies}, fh, indent=2)
    return rallies


def parse_corners(text):
    return [tuple(float(v) for v in pair.split(",")) for pair in text.split()]


def fmt(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--corners", required=True,
                        help='4 "x,y" fractions: far-left far-right near-right near-left')
    parser.add_argument("--work-dir", default=None, help="default: worker/output/<video name>")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--batch-size", type=int, default=4, help="4 on the Mac, 16 on the RTX 2070")
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.video))[0]
    work_dir = args.work_dir or os.path.join(WORKER_DIR, "output", name)
    last = [-1]

    def show_progress(frac):
        pct = int(frac * 100)
        if pct // 5 != last[0] // 5:
            print(f"  tracking {pct}%", flush=True)
            last[0] = pct

    rallies = process_video(args.video, parse_corners(args.corners), work_dir,
                            ckpt_path=args.ckpt, batch_size=args.batch_size, on_progress=show_progress)
    for i, r in enumerate(rallies, 1):
        print(f"Rally {i:>3}: {fmt(r['start'])} - {fmt(r['end'])}  ({r['duration']:5.1f}s)")
    print(f"\n{len(rallies)} rallies. Outputs in {work_dir}")


if __name__ == "__main__":
    main()
