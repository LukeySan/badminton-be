"""
Shuttle tracking with TrackNetV3 in "fast mode", callable from Python.

Ported from TrackNetV3's predict.py, keeping only the path we use in the product:
  --eval_mode nonoverlap   each frame goes through the network once (~9x faster
                           than the default overlapping windows)
  --large_video            frames are streamed from disk instead of loaded into RAM
  no InpaintNet            it gives garbage in nonoverlap mode

The other files in this folder (model.py, dataset.py, utils/) are copied unchanged
from TrackNetV3 (MIT, see LICENSE). They import each other as top-level modules
("from model import ..."), so this folder is put on sys.path below instead of
editing them.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import Video_IterableDataset
from utils.general import HEIGHT, WIDTH, get_model, to_img, to_img_format, write_pred_csv

# How many frames the background (median) image is built from. TrackNet's own
# version seeks to ~1800 frames and takes the median of the full-size stack: on
# match 4 (7 min, 720p) that took 124 s and 15 GB of memory, more than half as long
# as tracking itself. 300 frames read front to back: 10 s, 2.5 GB, and the same
# rallies and score on match 4. Fewer frames started to cost accuracy.
MEDIAN_SAMPLES = 300


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_location(heatmap):
    """Bounding box (x, y, w, h) of the largest blob in a binary heatmap (from test.py)."""
    if np.amax(heatmap) == 0:
        return 0, 0, 0, 0
    cnts, _ = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in cnts]
    return max(rects, key=lambda r: r[2] * r[3])


def heatmaps_to_coords(indices, y_pred, img_scaler):
    """Turn a batch of predicted heatmaps into per-frame shuttle coordinates (predict.py's predict())."""
    out = {"Frame": [], "X": [], "Y": [], "Visibility": []}
    indices = indices.numpy()
    y_pred = to_img_format((y_pred > 0.5).numpy())  # (N, L, H, W)

    prev_f_i = -1
    for n in range(indices.shape[0]):
        for f in range(indices.shape[1]):
            f_i = indices[n][f][1]
            if f_i == prev_f_i:
                break  # padding at the end of the video repeats the last frame
            x, y, w, h = predict_location(to_img(y_pred[n][f]))
            # Round to heatmap pixels first, then scale (same as predict.py, so outputs match exactly)
            cx, cy = int(int(x + w / 2) * img_scaler[0]), int(int(y + h / 2) * img_scaler[1])
            out["Frame"].append(int(f_i))
            out["X"].append(cx)
            out["Y"].append(cy)
            out["Visibility"].append(0 if cx == 0 and cy == 0 else 1)
            prev_f_i = f_i
    return out


def background_median(video_path, video_len, bg_mode, samples=MEDIAN_SAMPLES):
    """Median of ~`samples` evenly spaced frames: the empty-court background the model
    compares each frame against.

    Reads the video once, front to back, keeping every k-th frame. Seeking to each
    frame instead (as TrackNet does) makes H.264 decode from the previous keyframe
    every time. Post-processing is the same as dataset.Video_IterableDataset.__gen_median__.
    """
    step = max(1, video_len // samples)
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        for i in range(video_len):
            if i % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            elif not cap.grab():  # decode, but skip converting the frame
                break
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"Couldn't read any frames from {video_path}")
    median = np.median(frames, 0)[..., ::-1]  # BGR to RGB
    if bg_mode == "concat":
        median = Image.fromarray(median.astype("uint8"))
        median = np.array(median.resize(size=(WIDTH, HEIGHT)))
        median = np.moveaxis(median, -1, 0)
    return median


def track_shuttle(video_path, ckpt_path, out_csv, batch_size=4, device=None, on_progress=None):
    """Run TrackNet on every frame of video_path and write a Frame,Visibility,X,Y CSV
    (pixel coords of that video). on_progress(done_frames, total_frames) is called per batch."""
    device = device or pick_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    seq_len = ckpt["param_dict"]["seq_len"]
    bg_mode = ckpt["param_dict"]["bg_mode"]
    model = get_model("TrackNet", seq_len, bg_mode).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Passing a median skips the dataset's own slow one; the real one is set right
    # after, once the dataset has counted the frames.
    dataset = Video_IterableDataset(
        video_path, seq_len=seq_len, sliding_step=seq_len, bg_mode=bg_mode, median=np.zeros(1)
    )
    if bg_mode:
        dataset.median = background_median(video_path, dataset.video_len, bg_mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    img_scaler = (dataset.w / WIDTH, dataset.h / HEIGHT)

    pred = {"Frame": [], "X": [], "Y": [], "Visibility": []}
    for i, x in loader:
        with torch.no_grad():
            y_pred = model(x.float().to(device)).detach().cpu()
        for key, values in heatmaps_to_coords(i, y_pred, img_scaler).items():
            pred[key].extend(values)
        if on_progress:
            on_progress(len(pred["Frame"]), dataset.video_len)

    write_pred_csv(pred, save_file=out_csv)
    return out_csv
