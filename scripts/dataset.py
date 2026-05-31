# scripts/dataset.py

from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from scripts.utils import mask_to_bbox, jitter_bbox


class COD10KDataset(Dataset):
    def __init__(
        self,
        root_dir,
        is_train=False,
        bbox_jitter=0.0,
        max_samples=None,
        resize_to=(256, 256),   
    ):
        self.root = Path(root_dir)
        self.image_dir = self.root / "Image"
        self.gt_dir = self.root / "GT_Object"

        self.is_train = is_train
        self.bbox_jitter = bbox_jitter
        self.resize_to = resize_to

        self.image_files = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        if max_samples is not None:
            self.image_files = self.image_files[:max_samples]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        gt_path = self.gt_dir / f"{img_path.stem}.png"

        # --- load image ---
        image = Image.open(img_path).convert("RGB")

        if self.resize_to is not None:
            image = image.resize(self.resize_to, Image.BILINEAR)

        image_np = np.array(image).astype(np.float32) / 255.0   # H,W,3

        # --- load GT mask ---
        gt = Image.open(gt_path).convert("L")

        if self.resize_to is not None:
            gt = gt.resize(self.resize_to, Image.NEAREST)

        gt_np = (np.array(gt) > 0).astype(np.float32)           # H,W

        # --- generate bbox prompt from resized GT_Object ---
        bbox = mask_to_bbox(gt_np)

        h, w = gt_np.shape
        if self.is_train and self.bbox_jitter > 0:
            bbox = jitter_bbox(bbox, h=h, w=w, jitter_ratio=self.bbox_jitter)

        # --- convert to tensor ---
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)   # 3,H,W
        gt_tensor = torch.from_numpy(gt_np).unsqueeze(0)             # 1,H,W
        bbox_tensor = torch.from_numpy(bbox).float()                 # 4

        return {
            "image": image_tensor,
            "mask": gt_tensor,
            "bbox": bbox_tensor,
            "image_id": img_path.stem,
        }