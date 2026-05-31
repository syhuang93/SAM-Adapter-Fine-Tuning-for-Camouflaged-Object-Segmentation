import math
import torch
import torch.nn as nn

from segment_anything import sam_model_registry

class Adapter(nn.Module):
    def __init__(self, dim, bottleneck=64, zero_init=True):
      super().__init__()
      self.down = nn.Linear(dim, bottleneck)
      self.act = nn.GELU()
      self.up = nn.Linear(bottleneck, dim)
      self.zero_init = zero_init
      
      self._init_weights(self.zero_init)

    def _init_weights(self, zero_init):
      if zero_init:
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
      else:
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
      return self.up(self.act(self.down(x)))

class BlockWithAdapter(nn.Module):
    def __init__(self, orig_block, adapter_bottleneck=64, zero_init=True):
      super().__init__()
      self.block = orig_block

      dim = orig_block.mlp.lin2.out_features

      self.adapter = Adapter(dim, adapter_bottleneck, zero_init=zero_init)

    def forward(self, x):
      x = self.block(x)
      x = x + self.adapter(x)
      return x
  
def  build_sam_adapter(
      model_type='vit_b', 
      checkpoint=None, 
      bottleneck=64, 
      topk=None,
      zero_init=True,
      device='cuda'
):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    encoder = sam.image_encoder

    n_blocks = len(encoder.blocks)
    layer_indices = list(range(n_blocks)) if topk is None else list(range(n_blocks - topk, n_blocks))

    for i in layer_indices:
        encoder.blocks[i] = BlockWithAdapter(encoder.blocks[i], bottleneck, zero_init=zero_init)

    for param in sam.parameters():
        param.requires_grad = False

    for i in layer_indices:
        for param in encoder.blocks[i].adapter.parameters():
            param.requires_grad = True

    sam.to(device)

    return sam