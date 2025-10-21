from __future__ import annotations
from typing import List

import torch
import torch.nn as nn


class TeacherDinoV3(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        from transformers import AutoModel
        self.vit = AutoModel.from_pretrained(
            model_dir, local_files_only=True, output_hidden_states=True
        )
        self.embed_dim = self.vit.config.hidden_size
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward_tokens(self, x_rgb: torch.Tensor, tap_blocks_0based: List[int]):
        out = self.vit(x_rgb, output_hidden_states=True)
        hs = out.hidden_states
        tokens_list = []
        for blk in tap_blocks_0based:
            h = hs[blk + 1]
            tokens_list.append(h[:, 1:, :])
        return tokens_list


