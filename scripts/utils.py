# scripts/utils.py

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mask_to_bbox(mask: np.ndarray):
    """
    mask: H x W, binary mask (0/1)
    return: [x1, y1, x2, y2]
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape
        return np.array([0, 0, w - 1, h - 1], dtype=np.float32)

    x1 = xs.min()
    x2 = xs.max()
    y1 = ys.min()
    y2 = ys.max()

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def jitter_bbox(bbox, h, w, jitter_ratio=0.1):
    """
    Slightly perturb bbox during training.
    bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    dx = bw * jitter_ratio
    dy = bh * jitter_ratio

    x1 = np.clip(x1 - np.random.uniform(0, dx), 0, w - 1)
    y1 = np.clip(y1 - np.random.uniform(0, dy), 0, h - 1)
    x2 = np.clip(x2 + np.random.uniform(0, dx), 0, w - 1)
    y2 = np.clip(y2 + np.random.uniform(0, dy), 0, h - 1)

    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)

    return np.array([x1, y1, x2, y2], dtype=np.float32)