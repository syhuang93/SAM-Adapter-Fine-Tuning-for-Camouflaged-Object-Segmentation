# eval.py
import os
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from scripts.config import TEST_DIR, SAM_B_CHECKPOINT, OUTPUT_DIR
from scripts.dataset import COD10KDataset
from scripts.adapter import build_sam_adapter
from scripts.model_lora import build_sam_b_with_lora
from train import forward_sam_with_box


def _fmeasure_at_threshold(pred: np.ndarray, gt: np.ndarray,
                            threshold: float, beta_sq: float = 0.3) -> float:
    pred_bin = (pred >= threshold).astype(np.float64)
    gt_bin = (gt > 0.5).astype(np.float64)
    eps = 1e-7
    tp = (pred_bin * gt_bin).sum()
    fp = (pred_bin * (1 - gt_bin)).sum()
    fn = ((1 - pred_bin) * gt_bin).sum()
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    return float((1 + beta_sq) * p * r / (beta_sq * p + r + eps))


def _emeasure_at_threshold(pred: np.ndarray, gt: np.ndarray,
                            threshold: float) -> float:
    pred_bin = (pred >= threshold).astype(np.float64)
    gt_bin = (gt >  0.5).astype(np.float64)
    if gt_bin.max() == 0:
        return float(1.0 - pred_bin.mean())
    if gt_bin.min() == 1:
        return float(pred_bin.mean())
    ap = pred_bin - pred_bin.mean()
    ag = gt_bin - gt_bin.mean()
    align = (2 * ap * ag) / (ap**2 + ag**2 + 1e-9)
    enhanced = ((1 + align) / 2) ** 2
    return float(enhanced.mean())

def compute_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.abs(pred - (gt > 0.5).astype(np.float64)).mean())

def _s_object(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    mu, v = x.mean(), x.var()
    return float(2.0 * mu / (mu**2 + v + 1.0 + 1e-8))

def _ssim_quad(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1:
        return 1.0 if abs(float(x.mean()) - float(y.mean())) < 1e-6 else 0.0
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    C1, C2 = 1e-4, 1e-4
    num = (2*mx*my + C1) * (2*cov + C2)
    den = (mx**2 + my**2 + C1) * (vx + vy + C2)
    return float(num / (den + 1e-12))


def compute_smeasure(pred: np.ndarray, gt: np.ndarray,
                     alpha: float = 0.5) -> float:
    pred = pred.astype(np.float64)
    gt_bin = (gt > 0.5).astype(np.float64)
    y = gt_bin.mean()
    if y == 0:
        return float(1.0 - pred.mean())
    if y == 1:
        return float(pred.mean())

    fg = gt_bin > 0.5
    So = y * _s_object(pred[fg]) + (1 - y) * _s_object(1 - pred[~fg])

    H, W = pred.shape
    ys, xs = np.where(fg)
    cy = int(np.clip(round(float(ys.mean())), 1, H - 1))
    cx = int(np.clip(round(float(xs.mean())), 1, W - 1))
    quads = [
        (pred[:cy, :cx].ravel(), gt_bin[:cy, :cx].ravel()),
        (pred[:cy, cx:].ravel(), gt_bin[:cy, cx:].ravel()),
        (pred[cy:, :cx].ravel(), gt_bin[cy:, :cx].ravel()),
        (pred[cy:, cx:].ravel(), gt_bin[cy:, cx:].ravel()),
    ]
    total = float(H * W)
    Sr = sum((p.size / total) * _ssim_quad(p, g)
             for p, g in quads if p.size > 0)
    return float(max(0.0, alpha * So + (1 - alpha) * Sr))


def compute_weighted_fmeasure(pred: np.ndarray, gt: np.ndarray,
                               beta_sq: float = 1.0) -> float:
    pred = pred.astype(np.float64)
    gt_bin = (gt > 0.5).astype(np.float32)
    eps = 1e-7

    dist_fg = distance_transform_edt(gt_bin)
    dist_bg = distance_transform_edt(1.0 - gt_bin)
    dst = dist_fg + dist_bg
    W = 1.0 - dst / (dst.max() + eps)

    t = min(2.0 * pred.mean(), 1.0)
    pred_bin = (pred >= t).astype(np.float64)
    gt_f = gt_bin.astype(np.float64)

    WTP = (W * pred_bin * gt_f).sum()
    WFP = (W * pred_bin * (1 - gt_f)).sum()
    WFN = (W * (1 - pred_bin) * gt_f).sum()
    p = WTP / (WTP + WFP + eps)
    r = WTP / (WTP + WFN + eps)
    return float((1 + beta_sq) * p * r / (beta_sq * p + r + eps))


# --- F-measure variants ---

def compute_fmeasure_adaptive(pred, gt, beta_sq=0.3):
    return _fmeasure_at_threshold(pred, gt,
                                  threshold=min(2.0 * pred.mean(), 1.0),
                                  beta_sq=beta_sq)


def compute_fmeasure_mean(pred, gt, beta_sq=0.3):
    return _fmeasure_at_threshold(pred, gt,
                                  threshold=float(pred.mean()),
                                  beta_sq=beta_sq)


def compute_fmeasure_max(pred, gt, beta_sq=0.3, n_thresholds=255):
    thresholds = np.linspace(0, 1, n_thresholds)
    return float(max(
        _fmeasure_at_threshold(pred, gt, t, beta_sq) for t in thresholds
    ))

def compute_emeasure_adaptive(pred, gt):
    pred = pred.astype(np.float64)
    gt_bin = (gt > 0.5).astype(np.float64)
    if gt_bin.max() == 0:
        return float(1.0 - pred.mean())
    if gt_bin.min() == 1:
        return float(pred.mean())
    ap = pred   - pred.mean()
    ag = gt_bin - gt_bin.mean()
    align = (2 * ap * ag) / (ap**2 + ag**2 + 1e-9)
    enhanced = ((1 + align) / 2) ** 2
    return float(enhanced.mean())


def compute_emeasure_mean(pred, gt):
    return _emeasure_at_threshold(pred, gt, threshold=float(pred.mean()))


def compute_emeasure_max(pred, gt, n_thresholds=255):
    thresholds = np.linspace(0, 1, n_thresholds)
    return float(max(
        _emeasure_at_threshold(pred, gt, t) for t in thresholds
    ))

def save_visual(image_tensor, gt_tensor, pred_prob, image_id, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt_np = (gt_tensor[0].cpu().numpy() * 255).astype(np.uint8)
    pred_np = (pred_prob * 255).astype(np.uint8)
    H, W = img_np.shape[:2]
    canvas = Image.new("RGB", (W * 3, H))
    canvas.paste(Image.fromarray(img_np), (0,     0))
    canvas.paste(Image.fromarray(gt_np).convert("RGB"), (W, 0))
    canvas.paste(Image.fromarray(pred_np).convert("RGB"), (W * 2, 0))
    canvas.save(save_dir / f"{image_id}.png")

@torch.no_grad()
def run_evaluation(model, loader, device,
                   save_visuals=False, visual_dir=None):
    model.eval()

    scores = {k: [] for k in
              ["sm", "wfb", "mae",
               "fad", "fmn", "fmx",
               "ead", "emn", "emx"]}

    for batch in tqdm(loader, desc="Evaluating"):
        image = batch["image"][0].to(device)
        gt_mask = batch["mask"][0].unsqueeze(0).to(device)   # [1,1,H,W]
        bbox = batch["bbox"][0].to(device)
        img_id = batch["image_id"][0]

        pred_logits = forward_sam_with_box(
            sam_model=model,
            image=image,
            bbox=bbox,
            original_mask_shape=gt_mask.shape[-2:],
        )

        pred = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()
        gt = gt_mask[0, 0].cpu().numpy()

        scores["sm"].append(compute_smeasure(pred, gt))
        scores["wfb"].append(compute_weighted_fmeasure(pred, gt))
        scores["mae"].append(compute_mae(pred, gt))

        scores["fad"].append(compute_fmeasure_adaptive(pred, gt))
        scores["fmn"].append(compute_fmeasure_mean(pred, gt))
        scores["fmx"].append(compute_fmeasure_max(pred, gt))

        scores["ead"].append(compute_emeasure_adaptive(pred, gt))
        scores["emn"].append(compute_emeasure_mean(pred, gt))
        scores["emx"].append(compute_emeasure_max(pred, gt))

        if save_visuals and visual_dir:
            save_visual(image, gt_mask[0], pred, img_id, visual_dir)

    return {k: float(np.mean(v)) for k, v in scores.items()} | \
           {"n_samples": len(scores["sm"])}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str, default=SAM_B_CHECKPOINT)
    parser.add_argument("--model-type",   type=str, default="adapter",
                        choices=["adapter", "lora"])
    parser.add_argument("--test-dir",     type=str, default=TEST_DIR)
    parser.add_argument("--max-samples",  type=int, default=None)
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--visual-dir",   type=str,
                        default=os.path.join(OUTPUT_DIR, "eval_visuals"))
    parser.add_argument("--lora-r",       type=int, default=8)
    parser.add_argument("--lora-alpha",   type=int, default=16)
    parser.add_argument("--lora-blocks",  type=int, nargs="+", default=[10, 11])
    parser.add_argument("--bottleneck",   type=int, default=64)
    parser.add_argument("--topk", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.model_type == "adapter":
        model = build_sam_adapter(
            checkpoint=SAM_B_CHECKPOINT,
            bottleneck=args.bottleneck,
            device=device,
            topk=args.topk
        )
    else:
        model = build_sam_b_with_lora(
            checkpoint_path=SAM_B_CHECKPOINT,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_block_indices=args.lora_blocks,
            device=device,
        )

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt       = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    if "epoch" in ckpt: print(f"  Epoch     : {ckpt['epoch']}")
    if "eval_loss" in ckpt: print(f"  Eval loss : {ckpt['eval_loss']:.4f}")

    dataset = COD10KDataset(
        root_dir=args.test_dir, is_train=False, max_samples=args.max_samples
    )
    print(f"Test samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    r = run_evaluation(
        model=model, loader=loader, device=device,
        save_visuals=args.save_visuals,
        visual_dir=args.visual_dir if args.save_visuals else None,
    )

    print()
    print("=" * 48)
    print(f"  Results  ({r['n_samples']} samples)")
    print("=" * 48)
    print(f"  Sα    (structure measure)   : {r['sm']:.4f}")
    print(f"  Fβw   (weighted F-measure)  : {r['wfb']:.4f}")
    print(f"  M     (MAE)                 : {r['mae']:.4f}  ↓")
    print("-" * 48)
    print(f"  Ead   (E adaptive)          : {r['ead']:.4f}")
    print(f"  Emn   (E mean thr.)         : {r['emn']:.4f}")
    print(f"  Emx   (E maximum)           : {r['emx']:.4f}")
    print("-" * 48)
    print(f"  Fad   (F adaptive, β²=0.3)  : {r['fad']:.4f}")
    print(f"  Fmn   (F mean thr., β²=0.3) : {r['fmn']:.4f}")
    print(f"  Fmx   (F maximum,  β²=0.3)  : {r['fmx']:.4f}")
    print("=" * 48)
    print("  ↑ higher is better for all except M (MAE)")

    if args.save_visuals:
        print(f"\nVisuals saved to: {args.visual_dir}")
        
if __name__ == "__main__":
    main()