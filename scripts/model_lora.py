# scripts/model_lora.py

import math
import torch
import torch.nn as nn

from segment_anything import sam_model_registry


class LoRALinear(nn.Module):
    """
    Minimal LoRA wrapper for nn.Linear
    y = base(x) + scale * (x @ A^T @ B^T)
    """
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()

        assert isinstance(base_layer, nn.Linear), "LoRALinear only supports nn.Linear"

        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA parameters
        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))

        # initialization
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # freeze original linear
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.A.t()) @ self.B.t()
        return base_out + self.scale * lora_out


def inject_lora_into_sam_image_encoder(
    sam_model,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules=("qkv",),
    target_block_indices=None,
):
    """
    Inject LoRA into SAM image encoder attention layers.

    target_modules can include:
        - "qkv"
        - "proj"

    target_block_indices:
        list of block indices to inject LoRA into
        e.g. [10, 11] for the last two blocks
    """
    replaced = 0

    if target_block_indices is None:
        target_block_indices = list(range(len(sam_model.image_encoder.blocks)))

    for i, blk in enumerate(sam_model.image_encoder.blocks):
        if i not in target_block_indices:
            continue

        if "qkv" in target_modules and hasattr(blk.attn, "qkv") and isinstance(blk.attn.qkv, nn.Linear):
            blk.attn.qkv = LoRALinear(blk.attn.qkv, r=r, alpha=alpha, dropout=dropout)
            replaced += 1

        if "proj" in target_modules and hasattr(blk.attn, "proj") and isinstance(blk.attn.proj, nn.Linear):
            blk.attn.proj = LoRALinear(blk.attn.proj, r=r, alpha=alpha, dropout=dropout)
            replaced += 1

    return sam_model, replaced


def set_trainable_parameters(
    sam_model,
    train_mask_decoder: bool = False,
    train_prompt_encoder: bool = False,
):
    """
    Freeze everything first, then only unfreeze:
      - LoRA params
      - optional mask decoder
      - optional prompt encoder
    """
    # freeze all
    for p in sam_model.parameters():
        p.requires_grad = False

    # unfreeze LoRA params
    for name, p in sam_model.named_parameters():
        if name.endswith(".A") or name.endswith(".B"):
            p.requires_grad = True

    if train_mask_decoder:
        for p in sam_model.mask_decoder.parameters():
            p.requires_grad = True

    if train_prompt_encoder:
        for p in sam_model.prompt_encoder.parameters():
            p.requires_grad = True


def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_sam_b_with_lora(
    checkpoint_path: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules=("qkv",),
    target_block_indices=None,
    train_mask_decoder: bool = False,
    train_prompt_encoder: bool = False,
    device: str = "cuda",
):
    """
    Build SAM-B + inject LoRA + set trainable params
    """

    # 1) load original SAM on CPU first
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)

    # 2) inject LoRA
    sam, replaced = inject_lora_into_sam_image_encoder(
        sam_model=sam,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=target_modules,
        target_block_indices=target_block_indices,
    )

    # 3) set trainable params
    set_trainable_parameters(
        sam_model=sam,
        train_mask_decoder=train_mask_decoder,
        train_prompt_encoder=train_prompt_encoder,
    )

    # 4) move everything to device
    sam.to(device)

    trainable, total = count_trainable_parameters(sam)

    print(f"LoRA injected into {replaced} layers")
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    return sam