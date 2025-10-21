from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
import timm


def tokens_from_cnn_bottleneck(bottleneck: torch.Tensor) -> torch.Tensor:
    b, c, h, w = bottleneck.shape
    return bottleneck.permute(0, 2, 3, 1).reshape(b, h * w, c)


class TinyViTWithTaps(nn.Module):
    def __init__(self, name: str = "vit_tiny_patch16_224", pretrained: bool = False):
        super().__init__()
        self.vit = timm.create_model(name, pretrained=pretrained)
        self.embed_dim = self.vit.embed_dim
        self.handles: List = []
        self.tokens: Dict[int, torch.Tensor] = {}

    def register_taps(self, tap_blocks_0based: List[int]):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.tokens.clear()
        for i, blk in enumerate(self.vit.blocks):
            if i in set(tap_blocks_0based):
                self.handles.append(blk.register_forward_hook(self._hook_block(i)))

    def _hook_block(self, idx: int):
        def fn(module, inp, out):
            self.tokens[idx] = out[:, 1:, :]
        return fn

    def forward_tokens(self, x: torch.Tensor, tap_blocks_0based: List[int]) -> List[torch.Tensor]:
        self.register_taps(tap_blocks_0based)
        self.tokens.clear()
        _ = self.vit(x)
        return [self.tokens[i] for i in tap_blocks_0based]


