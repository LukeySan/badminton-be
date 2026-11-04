"""
Rally detection, built on TrackNetV3's shuttle CSV.

A rally is a stretch of time where the shuttle is MOVING over your court.
Just "visible" isn't enough: between points the shuttle is still on screen,
lying on the floor or held by the server, so we look at how fast it moves.
Short pauses (shuttle briefly hidden or missed) don't end a rally; a long
pause (the point is over) does.

With --court (from pick_court.py), detections outside your court are ignored,
so play on neighboring courts doesn't stretch your rallies. The zone is your
court plus the air straight above it, since a high shuttle appears above the
court lines in the image.

Usage:
    python rally_detect.py prediction/clipped_vid_ball.csv ../clipped_vid.mp4 --court court.json
    python rally_detect.py <csv> <video> --court court.json --timeline     # per-second debug view
    python rally_detect.py <csv> <video> --court court.json --min-speed 0.01 --gap 2.5
"""

import argparse
import json

import cv2
import numpy as np
import pandas as pd


def load_corners(court_file):
    with open(court_file) as fh:
        c = json.load(fh)["corners"]
    return c["far-left"], c["far-right"], c["near-right"], c["near-left"]


def court_zone(corners):
    """Polygon covering the court and the air above it, in video pixels."""
    fl, fr, nr, nl = corners
    # Extend the far corners straight up to the top of the frame for high shots.
    poly = [(fl[0], 0), (fr[0], 0), fr, nr, nl, fl]
    return np.array(poly, dtype=np.float32)


def in_zone(zone, x, y, margin):
    # Positive distance = inside; allow a small margin outside the lines.
    return cv2.pointPolygonTest(zone, (float(x), float(y)), True) >= -margin


def find_hits(xs, ys, vis, s, e, min_move, fps, k=3, min_angle=100):
    """Approximate hits: frames where the shuttle's direction changes sharply.

    A hit reverses the shuttle's path abruptly; the top of a high arc only
    bends gradually, so a sharp-angle test separates the two reasonably well.
    """
    hits = []
    min_sep = int(fps * 0.35)                    # two hits can't be closer than ~0.35s
    for i in range(max(s, k), min(e, len(xs) - k - 1) + 1):
        if not (vis[i] and vis[i - k] and vis[i + k]):
            continue
        v_in = np.array([xs[i] - xs[i - k], ys[i] - ys[i - k]]) / k
        v_out = np.array([xs[i + k] - xs[i], ys[i + k] - ys[i]]) / k
        n_in, n_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
        if n_in < min_move or n_out < min_move:
            continue
        cos = np.dot(v_in, v_out) / (n_in * n_out)
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        if angle >= min_angle:
            if hits and i - hits[-1][0] < min_sep:
                if angle > hits[-1][1]:
                    hits[-1] = (i, angle)        # keep the sharpest turn in a cluster
            else:
                hits.append((i, angle))
    return [h[0] for h in hits]


def fmt(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


def rule_detect(csv_path, video_path, court_path=None, margin=40, min_speed=0.005, gap=2.0,
                still=0.4, still_speed=0.35, min_length=2.0):
    """Run the hand-written rules and return per-frame signals plus rally segments.
    Used by this script and, as extra features, by rally_model.py (the hybrid)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    if not fps or fps <= 0:
        raise SystemExit(f"Couldn't read the frame rate of {video_path}; check the video path.")

    df = pd.read_csv(csv_path).sort_values("Frame").reset_index(drop=True)
    total = len(df)
    vis_raw = (df["Visibility"] == 1).to_numpy()
    xs, ys = df["X"].to_numpy(float), df["Y"].to_numpy(float)

    # 1) Court filter
    vis = vis_raw
    if court_path:
        corners = load_corners(court_path)
        zone = court_zone(corners)
        vis = np.array([v and in_zone(zone, x, y, margin) for v, x, y in zip(vis_raw, xs, ys)])
        nl, nr = np.array(corners[3], float), np.array(corners[2], float)
        scale = np.linalg.norm(nr - nl)          # near baseline width in pixels
    else:
        scale = video_w * 0.5                    # rough guess without a court

    # 2) Speed: pixels moved since the previous detection (within a few frames), per frame
    speed = np.full(total, np.nan)
    last = None
    for i in range(total):
        if not vis[i]:
            continue
        if last is not None and i - last <= 3:
            speed[i] = np.hypot(xs[i] - xs[last], ys[i] - ys[last]) / (i - last)
        last = i
    # Median over ~0.2s smooths out single-frame jitter and single-frame jumps
    win = max(3, int(fps * 0.2)) | 1
    smooth = pd.Series(speed).rolling(win, center=True, min_periods=2).median().to_numpy()
    threshold = min_speed * scale
    moving = np.nan_to_num(smooth, nan=0.0) > threshold

    # 3) "Landed" frames: the shuttle is SEEN and nearly motionless for a while.
    #    In a real rally the shuttle never sits still, so a short stillness means it
    #    hit the floor; whatever comes after (pickup, hand-back) is not this rally.
    #    Frames where the shuttle is simply not detected don't count as still.
    is_still = vis & (np.nan_to_num(smooth, nan=np.inf) < still_speed * threshold)
    min_run = max(1, int(still * fps))
    landed = np.zeros(total, dtype=bool)
    run_start = None
    for i in range(total + 1):
        if i < total and is_still[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            if i - run_start >= min_run:
                landed[run_start:i] = True
            run_start = None
    landed_cum = np.concatenate([[0], np.cumsum(landed)])

    # 4) Group moving frames into rallies; a rally ends after a pause longer than
    #    gap, OR as soon as the shuttle has landed (if still > 0).
    gap_frames = gap * fps
    segs, start, prev = [], None, None
    for f in np.flatnonzero(moving):
        if start is None:
            start = prev = f
        elif f - prev > gap_frames or (still > 0 and landed_cum[f] - landed_cum[prev + 1] > 0):
            segs.append((start, prev))
            start = f
        prev = f
    if start is not None:
        segs.append((start, prev))
    segs = [(s, e) for s, e in segs if (e - s) / fps >= min_length]

    return {"fps": fps, "total": total, "vis_raw": vis_raw, "vis": vis, "xs": xs, "ys": ys,
            "smooth": smooth, "threshold": threshold, "scale": scale,
            "moving": moving, "landed": landed, "segments": segs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="TrackNetV3 prediction CSV (Frame, Visibility, X, Y)")
    parser.add_argument("video", help="the original video, used to read the frame rate")
    parser.add_argument("--court", default=None, help="court.json from pick_court.py")
    parser.add_argument("--margin", type=float, default=40,
                        help="pixels outside the court lines still counted as your court")
    parser.add_argument("--min-speed", type=float, default=0.005,
                        help="how fast the shuttle must move to count as 'in play', as a fraction "
                             "of the court's width per frame (raise it if pauses still count as play)")
    parser.add_argument("--gap", type=float, default=2.0,
                        help="seconds of no movement before a rally is considered over")
    parser.add_argument("--still", type=float, default=0.4,
                        help="end a rally once the shuttle is seen sitting still for this many "
                             "seconds (it landed), so pickups don't extend the rally; 0 turns it off")
    parser.add_argument("--still-speed", type=float, default=0.35,
                        help="'sitting still' means slower than this fraction of --min-speed")
    parser.add_argument("--min-length", type=float, default=2.0,
                        help="ignore 'rallies' shorter than this many seconds")
    parser.add_argument("--min-hits", type=int, default=0,
                        help="drop segments with fewer hits than this (2 drops most hand-backs)")
    parser.add_argument("--min-peak", type=float, default=0.0,
                        help="drop segments whose top speed (fraction of court width per frame) "
                             "is below this; hand-backs are slow")
    parser.add_argument("--timeline", action="store_true",
                        help="print a per-second view of what the script sees, for tuning")
    parser.add_argument("--out", default="rallies.json", help="where to save the results")
    args = parser.parse_args()

    r = rule_detect(args.csv, args.video, args.court, args.margin, args.min_speed, args.gap,
                    args.still, args.still_speed, args.min_length)
    fps, total, vis, xs, ys = r["fps"], r["total"], r["vis"], r["xs"], r["ys"]
    smooth, threshold, scale, moving = r["smooth"], r["threshold"], r["scale"], r["moving"]

    print(f"Frame rate: {fps:.2f} fps, {total} frames ({fmt(total / fps)})")
    print(f"Shuttle detected in {r['vis_raw'].sum()} frames ({100 * r['vis_raw'].mean():.0f}%)")
    if args.court:
        print(f"Court filter: ignored {int(r['vis_raw'].sum() - vis.sum())} detections outside your court")
    print(f"Moving fast enough to count as play: {moving.sum()} frames "
          f"({100 * moving.mean():.0f}%), threshold {threshold:.1f} px/frame\n")

    if args.timeline:
        print(" sec  detected  moving  median speed (px/frame)")
        for s in range(int(np.ceil(total / fps))):
            a, b = int(s * fps), min(int((s + 1) * fps), total)
            sp = np.nanmedian(smooth[a:b]) if np.any(~np.isnan(smooth[a:b])) else 0
            print(f"{fmt(s):>6}  {100 * vis[a:b].mean():5.0f}%  {100 * moving[a:b].mean():5.0f}%   "
                  f"{sp:6.1f}  {'#' * min(40, int(sp / max(threshold, 1e-6) * 4))}")
        print()

    # 5) Describe each segment: hits and top speed help tell real rallies from
    #    the shuttle being handed back to the server (one slow, soft flight).
    results, skipped = [], []
    for s, e in r["segments"]:
        duration = (e - s) / fps
        hits = find_hits(xs, ys, vis, s, e, min_move=0.5 * threshold, fps=fps)
        seg = smooth[s:e + 1]
        peak = float(np.nanmax(seg)) if np.any(~np.isnan(seg)) else 0.0
        peak_rel = peak / scale
        info = {
            "start": round(s / fps, 2),
            "end": round(e / fps, 2),
            "duration": round(duration, 2),
            "hits": len(hits),
            "hit_times": [round(h / fps, 2) for h in hits],
            "peak_speed": round(peak_rel, 4),
        }
        if len(hits) < args.min_hits or peak_rel < args.min_peak:
            skipped.append(info)
            continue
        info["id"] = len(results) + 1
        results.append(info)

    if not results:
        print("No rallies found. Try a lower --min-speed or a smaller --min-length.")
    for res in results:
        print(f"Rally {res['id']:>2}: {fmt(res['start'])} - {fmt(res['end'])}  ({res['duration']:4.1f}s)  "
              f"hits ~{res['hits']:>2}  peak speed {res['peak_speed']:.3f}")
    if skipped:
        print("\nSkipped (look like hand-backs, not rallies):")
        for res in skipped:
            print(f"         {fmt(res['start'])} - {fmt(res['end'])}  ({res['duration']:4.1f}s)  "
                  f"hits ~{res['hits']:>2}  peak speed {res['peak_speed']:.3f}")

    with open(args.out, "w") as fh:
        json.dump({"fps": fps, "rallies": results}, fh, indent=2)
    print(f"\nSaved {len(results)} rallies to {args.out}")


if __name__ == "__main__":
    main()