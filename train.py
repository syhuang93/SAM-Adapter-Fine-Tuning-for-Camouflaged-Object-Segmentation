# scripts/train_lora.py

import os
import gc
import csv
import json
import argparse
from pathlib import Path
import time
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from segment_anything.utils.transforms import ResizeLongestSide

from scripts.config import (
    TRAIN_DIR,
    EVAL_DIR,
    SAM_B_CHECKPOINT,
    OUTPUT_DIR,
    BATCH_SIZE,
    LR,
    EPOCHS,
    BBOX_JITTER,
    NUM_WORKERS,
)
from scripts.dataset import COD10KDataset
from scripts.model_lora import build_sam_b_with_lora
from scripts.adapter import build_sam_adapter


# -------------------------
# Loss functions
# -------------------------
def dice_loss(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


# -------------------------
# Helper: prepare one sample for SAM
# -------------------------
def prepare_sam_inputs(image_tensor, bbox_tensor, sam_model, device):
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    original_size = image_np.shape[:2]   # (H, W)

    resize_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    input_image = resize_transform.apply_image(image_np)

    input_image_torch = torch.as_tensor(input_image, device=device)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_image_torch = sam_model.preprocess(input_image_torch)

    box = resize_transform.apply_boxes(bbox_tensor[None, :].cpu().numpy(), original_size)
    box_torch = torch.as_tensor(box, dtype=torch.float, device=device)

    return input_image_torch, box_torch, original_size


# -------------------------
# Forward
# -------------------------
def forward_sam_with_box(sam_model, image, bbox, original_mask_shape):
    device = next(sam_model.parameters()).device

    input_image, box_torch, _ = prepare_sam_inputs(image, bbox, sam_model, device)

    image_embedding = sam_model.image_encoder(input_image)

    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )

    low_res_masks, _ = sam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    pred_mask_logits = F.interpolate(
        low_res_masks,
        size=original_mask_shape,
        mode="bilinear",
        align_corners=False,
    )

    return pred_mask_logits


# -------------------------
# Train / Eval loops
# -------------------------
def train_one_epoch(model, loader, optimizer, scheduler, bce_loss_fn, device, scaler):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        # print(f"[step {step}] batch fetched")
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        bboxes = batch["bbox"].to(device)

        image = images[0]
        gt_mask = masks[0].unsqueeze(0)   # [1,1,H,W]
        bbox = bboxes[0]

        # print(f"[step {step}] before forward")

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            pred_mask_logits = forward_sam_with_box(
                model,
                image=image,
                bbox=bbox,
                original_mask_shape=gt_mask.shape[-2:],
            )

            loss_bce = bce_loss_fn(pred_mask_logits, gt_mask)
            loss_dice = dice_loss(pred_mask_logits, gt_mask)
            loss = loss_bce + loss_dice

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        running_loss += loss.item()

        # print(f"[step {step}] after forward")

    return running_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, bce_loss_fn, device):
    model.eval()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        bboxes = batch["bbox"].to(device)

        image = images[0]
        gt_mask = masks[0].unsqueeze(0)
        bbox = bboxes[0]

        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            pred_mask_logits = forward_sam_with_box(
                model,
                image=image,
                bbox=bbox,
                original_mask_shape=gt_mask.shape[-2:],
            )

            loss_bce = bce_loss_fn(pred_mask_logits, gt_mask)
            loss_dice = dice_loss(pred_mask_logits, gt_mask)
            loss = loss_bce + loss_dice

        running_loss += loss.item()

    return running_loss / len(loader)

def save_loss_history(loss_history, out_dir, tag):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"loss_history_{tag}.csv"
    json_path = out_dir / f"loss_history_{tag}.json"
    plot_path = out_dir / f"loss_curve_{tag}.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", 
                "train_loss", 
                "eval_loss", 
                "lr",
                "train_time_sec",
                "eval_time_sec",
                "epoch_time_sec",
                "total_params",
                "trainable_params",
                "gpu_allocated_mb",
                "gpu_reserved_mb",
                "gpu_peak_allocated_mb",
                "gpu_peak_reserved_mb",
                "cpu_rss_mb",
            ],
        )
        writer.writeheader()
        for row in loss_history:
            writer.writerow(row)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(loss_history, f, ensure_ascii=False, indent=2)

    try:
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in loss_history]
        train_losses = [row["train_loss"] for row in loss_history]
        eval_losses = [row["eval_loss"] for row in loss_history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="train_loss")
        plt.plot(epochs, eval_losses, label="eval_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to save loss plot: {e}")
        plot_path = None

    return csv_path, json_path, plot_path

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_gpu_memory_mb():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
        return {
            "gpu_allocated_mb": allocated,
            "gpu_reserved_mb": reserved,
            "gpu_peak_allocated_mb": peak_allocated,
            "gpu_peak_reserved_mb": peak_reserved,
        }
    return {
        "gpu_allocated_mb": None,
        "gpu_reserved_mb": None,
        "gpu_peak_allocated_mb": None,
        "gpu_peak_reserved_mb": None,
    }

def get_cpu_memory_mb():
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / 1024**2
    return {"cpu_rss_mb": rss_mb}

# -------------------------
# Argument parser
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM-Adapter on COD10K")
    parser.add_argument("--bottleneck", type=int, default=64,
                        help="Adapter bottleneck dimension (default: 64)")
    parser.add_argument("--topk", type=int, default=None,
                        help="Number of last ViT blocks to inject adapters into. "
                             "None = all blocks (default: None)")
    parser.add_argument("--zero-init", action=argparse.BooleanOptionalAction, default=True,
                        help="Zero-initialize adapter up-projection (default: True). "
                             "Use --no-zero-init to disable.")
    return parser.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --- Adapter config ---
    bottleneck = args.bottleneck
    topk = args.topk
    zero_init = args.zero_init
    tag = f"topk{topk}" if topk is not None else "all"

    print(f"Adapter config: bottleneck={bottleneck}, topk={topk}, zero_init={zero_init}")

    train_dataset = COD10KDataset(
        TRAIN_DIR,
        is_train=True,
        bbox_jitter=BBOX_JITTER,
        max_samples=None
    )

    eval_dataset = COD10KDataset(
        EVAL_DIR,
        is_train=False,
        max_samples=None
    )

    print("Train dataset size:", len(train_dataset))
    print("Eval dataset size :", len(eval_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = build_sam_adapter(
        checkpoint=SAM_B_CHECKPOINT,
        bottleneck=bottleneck,
        topk=topk,
        zero_init=zero_init,
    )

    total_params, trainable_params = count_parameters(model)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Trainable ratio : {100 * trainable_params / total_params:.4f}%")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )

    total_steps = EPOCHS * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    bce_loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    loss_dir = os.path.join(OUTPUT_DIR, "loss_curves")
    os.makedirs(loss_dir, exist_ok=True)

    best_eval_loss = float("inf")
    best_ckpt_path = os.path.join(OUTPUT_DIR, f"best_adapter_{tag}_bn{bottleneck}.pth")
    loss_history = []

    total_start_time = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    global_peak_mem = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()

        train_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, bce_loss_fn, device, scaler)
        train_time = time.perf_counter() - train_start
        
        eval_start = time.perf_counter()
        eval_loss = evaluate(model, eval_loader, bce_loss_fn, device)
        eval_time = time.perf_counter() - eval_start

        epoch_time = time.perf_counter() - epoch_start
        mem_gpu = get_gpu_memory_mb()
        mem_cpu = get_cpu_memory_mb()

        if mem_gpu["gpu_peak_allocated_mb"] is not None:
            global_peak_mem = max(global_peak_mem, mem_gpu["gpu_peak_allocated_mb"])

        current_lr = optimizer.param_groups[0]["lr"]
        loss_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "eval_loss": float(eval_loss),
                "lr": float(current_lr),
                "train_time_sec": float(train_time),
                "eval_time_sec": float(eval_time),
                "epoch_time_sec": float(epoch_time),
                "total_params": int(total_params),
                "trainable_params": int(trainable_params),
                **mem_gpu,
                **mem_cpu,
            }
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Eval  Loss: {eval_loss:.4f}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eval_loss": eval_loss,
                },
                best_ckpt_path,
            )
            print(f"Saved best checkpoint to: {best_ckpt_path}")

    total_time = time.perf_counter() - total_start_time

    if torch.cuda.is_available():
        total_peak_mem = global_peak_mem
    else:
        total_peak_mem = None

    print("\n===== Training Summary =====")
    print(f"Total time: {total_time:.2f} sec")
    print(f"Peak GPU memory: {total_peak_mem:.2f} MB")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    csv_path, json_path, plot_path = save_loss_history(loss_history, loss_dir, tag)
    print("\nSaved loss history to:")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")
    if plot_path is not None:
        print(f"  PNG : {plot_path}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()