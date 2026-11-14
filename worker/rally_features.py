"""
Feature code for the v1.1 detector, copied from research (../badminton-test/TrackNetV3):

    frame_features()    rally_model.make_features() + add_rule_features() (no player
                        or YOLO feature groups; fps passed in instead of read from the video)
    to_rallies()        rally_model.to_rallies()
    segment_features()  two_stage.segment_table() without labels and player features
    refine()            two_stage.refine()

Inference only: the models are trained in research (train_v11.py) and loaded by
detector.py. If you change anything here, the shipped models no longer see the
features they were trained on: change it in research too, retrain, and check the
worker against research's reference output (worker/tests/test_detector_v11.py).
"""

import numpy as np
import pandas as pd

from rally_detect import court_zone, find_hits, in_zone, load_corners, rule_detect


def frame_features(csv_path, court_path, fps, rule_settings, margin=40):
    """One row of features per video frame, normalized by court size so they
    mean the same thing in any video resolution or camera distance."""
    df = pd.read_csv(csv_path).sort_values("Frame").reset_index(drop=True)
    xs, ys = df["X"].to_numpy(float), df["Y"].to_numpy(float)
    vis_tracknet = (df["Visibility"] == 1).to_numpy()

    corners = load_corners(court_path)
    fl, fr, nr, nl = [np.array(c, float) for c in corners]
    zone = court_zone(corners)
    vis = np.array([v and in_zone(zone, x, y, margin) for v, x, y in zip(vis_tracknet, xs, ys)])

    width = np.linalg.norm(nr - nl)                         # near baseline, px
    near_y, far_y = (nl[1] + nr[1]) / 2, (fl[1] + fr[1]) / 2
    height = max(near_y - far_y, 1.0)                       # court depth on screen, px
    center_x = (nl[0] + nr[0]) / 2

    # Position relative to the court: xn 0 = centre line, yn 0 = near baseline,
    # 1 = far baseline, >1 = in the air above the far half.
    xn = np.where(vis, (xs - center_x) / width, np.nan)
    yn = np.where(vis, (near_y - ys) / height, np.nan)

    n = len(df)
    vx = np.full(n, np.nan); vy = np.full(n, np.nan)
    last = None
    for i in range(n):
        if not vis[i]:
            continue
        if last is not None and i - last <= 3:
            vx[i] = (xs[i] - xs[last]) / (i - last) / width
            vy[i] = (ys[i] - ys[last]) / (i - last) / width
        last = i
    speed = np.hypot(vx, vy)

    F = pd.DataFrame({"vis": vis.astype(float), "xn": xn, "yn": yn,
                      "vx": vx, "vy": vy, "speed": speed})
    still = (vis & (np.nan_to_num(speed, nan=1.0) < 0.002)).astype(float)
    turn = np.zeros(n)                                     # sharp direction changes
    for i in range(3, n - 3):
        if vis[i - 3] and vis[i] and vis[i + 3]:
            a = np.array([xs[i] - xs[i - 3], ys[i] - ys[i - 3]])
            b = np.array([xs[i + 3] - xs[i], ys[i + 3] - ys[i]])
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0.01 * width and nb > 0.01 * width:
                turn[i] = float(np.dot(a, b) / (na * nb) < -0.17)   # > ~100 degrees
    F["still"], F["turn"] = still, turn

    # Context: the same signals summarized over windows around each frame.
    for sec in (0.25, 0.5, 1.0, 2.0):
        w = max(3, int(fps * sec)) | 1
        r = F.rolling(w, center=True, min_periods=1)
        F[f"vis_c{sec}"] = r["vis"].mean()
        F[f"spd_mean_c{sec}"] = r["speed"].mean()
        F[f"spd_max_c{sec}"] = r["speed"].max()
        F[f"spd_med_c{sec}"] = r["speed"].median()
        F[f"x_std_c{sec}"] = r["xn"].std()
        F[f"y_std_c{sec}"] = r["yn"].std()
        F[f"y_min_c{sec}"] = r["yn"].min()
        F[f"y_max_c{sec}"] = r["yn"].max()
        F[f"vy_mean_c{sec}"] = r["vy"].mean()
        F[f"still_c{sec}"] = r["still"].mean()
        F[f"turns_c{sec}"] = r["turn"].sum()

    # Looking only backward / only forward helps find where rallies start and end.
    rev = F[::-1]
    for sec in (1.0, 3.0):
        w = max(2, int(fps * sec))
        past = F.rolling(w, min_periods=1)
        fut = rev.rolling(w, min_periods=1)
        for col in ("speed", "vis", "still", "turn"):
            agg = "sum" if col == "turn" else "mean"
            F[f"{col}_past{sec}"] = getattr(past[col], agg)()
            F[f"{col}_fut{sec}"] = getattr(fut[col], agg)()[::-1].to_numpy()

    _add_rule_features(F, csv_path, court_path, margin, fps, rule_settings)
    return F


def _add_rule_features(F, csv_path, court_path, margin, fps, rule_settings):
    """Hybrid: give the model the hand-written rules' opinion, so it can learn
    when to trust them (they rarely miss rallies) and when to overrule them
    (pickups and tosses they mistake for play)."""
    r = rule_detect(csv_path, None, court_path, margin=margin, fps=fps, **rule_settings)
    n = len(F)
    in_rally = np.zeros(n)
    since_start = np.full(n, 10.0)   # seconds since the rules' rally began (capped)
    until_end = np.full(n, 10.0)     # seconds until the rules' rally ends (capped)
    seg_len = np.zeros(n)
    for s, e in r["segments"]:
        in_rally[s:e + 1] = 1
        idx = np.arange(s, e + 1)
        since_start[s:e + 1] = np.minimum((idx - s) / fps, 10)
        until_end[s:e + 1] = np.minimum((e - idx) / fps, 10)
        seg_len[s:e + 1] = (e - s) / fps
    landed = r["landed"].astype(float)
    since_landed = np.full(n, 10.0)
    last = None
    for i in range(n):
        if landed[i]:
            last = i
        if last is not None:
            since_landed[i] = min((i - last) / fps, 10)

    F["rule_in_rally"] = in_rally
    F["rule_moving"] = r["moving"].astype(float)
    F["rule_landed"] = landed
    F["rule_since_start"] = since_start
    F["rule_until_end"] = until_end
    F["rule_seg_len"] = seg_len
    F["rule_since_landed"] = since_landed
    for sec in (0.5, 2.0):
        w = max(3, int(fps * sec)) | 1
        F[f"rule_moving_c{sec}"] = F["rule_moving"].rolling(w, center=True, min_periods=1).mean()
        F[f"rule_landed_c{sec}"] = F["rule_landed"].rolling(w, center=True, min_periods=1).mean()


def to_rallies(prob, fps, threshold=0.5, smooth=0.4, merge_gap=0.7, min_len=1.5):
    """Frame probabilities -> rallies [(start, end)] in seconds."""
    w = max(3, int(fps * smooth)) | 1
    p = pd.Series(prob).rolling(w, center=True, min_periods=1).median().to_numpy()
    on = np.flatnonzero(p > threshold)
    segs, start, prev = [], None, None
    for f in on:
        if start is None:
            start = prev = f
        elif f - prev > merge_gap * fps:
            segs.append((start, prev)); start = f
        prev = f
    if start is not None:
        segs.append((start, prev))
    return [(round(s / fps, 2), round(e / fps, 2)) for s, e in segs if (e - s) / fps >= min_len]


def segment_features(F, r, prob, fps):
    """One row of features per candidate rally (rule_detect result r), and the
    candidates as (start, end) seconds."""
    segs = r["segments"]
    cands = [(round(s / fps, 2), round(e / fps, 2)) for s, e in segs]
    rows = []
    for k, (s, e) in enumerate(segs):
        sl = slice(s, e + 1)
        seg = F.iloc[sl]
        dur = (e - s) / fps
        smooth = r["smooth"][sl]
        p = prob[sl]
        rows.append({
            "duration": dur,
            "gap_before": min((s - segs[k - 1][1]) / fps, 60) if k else 60.0,
            "gap_after": min((segs[k + 1][0] - e) / fps, 60) if k + 1 < len(segs) else 60.0,
            "vis": seg["vis"].mean(),
            "speed_mean": seg["speed"].mean(), "speed_max": seg["speed"].max(),
            "speed_std": seg["speed"].std(),
            "peak_smooth": (np.nanmax(smooth) if np.any(~np.isnan(smooth)) else 0) / r["scale"],
            "still": seg["still"].mean(), "landed": r["landed"][sl].mean(),
            "moving": r["moving"][sl].mean(),
            "turns_per_s": seg["turn"].sum() / max(dur, 0.1),
            "hits": len(find_hits(r["xs"], r["ys"], r["vis"], s, e,
                                  min_move=0.5 * r["threshold"], fps=fps)),
            "y_range": seg["yn"].max() - seg["yn"].min(), "y_max": seg["yn"].max(),
            "x_std": seg["xn"].std(),
            "model_mean": p.mean(), "model_max": p.max(), "model_frac": (p > 0.5).mean(),
        })
    return pd.DataFrame(rows), cands


def refine(chapters, model_rallies, max_shift):
    """Take the overlapping model rally's start/end where it's within max_shift seconds."""
    out = []
    for s, e in chapters:
        best, best_ov = None, 0.0
        for ms, me in model_rallies:
            ov = min(e, me) - max(s, ms)
            if ov > best_ov:
                best, best_ov = (ms, me), ov
        if best:
            s = best[0] if abs(best[0] - s) <= max_shift else s
            e = best[1] if abs(best[1] - e) <= max_shift else e
        out.append((s, e))
    return out
