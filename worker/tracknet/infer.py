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
from torch.utils.data import DataLoader

from dataset import Video_IterableDataset
from utils.general import HEIGHT, WIDTH, get_model, to_img, to_img_format, write_pred_csv


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

    dataset = Video_IterableDataset(video_path, seq_len=seq_len, sliding_step=seq_len, bg_mode=bg_mode)
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
