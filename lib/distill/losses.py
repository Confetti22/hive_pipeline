from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureMimicCosine(nn.Module):
    def __init__(self, mode: str = "vit2vit"):
        super().__init__()
        self.mode = mode

    def _normalize(self, s: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_ = F.layer_norm(s, s.shape[-1:])
        t_ = F.layer_norm(t, t.shape[-1:])
        if self.mode == "vit2cnn":
            s_ = F.normalize(s_, dim=-1)
            t_ = F.normalize(t_, dim=-1)
        return s_, t_

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s, t = self._normalize(s, t.detach())
        cos = F.cosine_similarity(s, t, dim=-1)
        return (1.0 - cos).mean()


def _local_affinity_profile(x: torch.Tensor, anchors=64, window=7) -> torch.Tensor:
    b, n, c = x.shape
    h = w = int(math.sqrt(n)); assert h * w == n
    x = F.layer_norm(x, x.shape[-1:])
    x = F.normalize(x, dim=-1)
    idx = torch.randperm(n, device=x.device)[:min(anchors, n)]
    hh, ww = idx // w, idx % w
    rad = window // 2
    grid = (torch.arange(h, device=x.device)[:, None] * w + torch.arange(w, device=x.device)[None, :])
    sims = []
    for i in range(idx.numel()):
        h0, w0 = hh[i].item(), ww[i].item()
        hs = slice(max(0, h0 - rad), min(h, h0 + rad + 1))
        ws = slice(max(0, w0 - rad), min(w, w0 + rad + 1))
        neigh = grid[hs, ws].reshape(-1)
        anc = x[:, h0 * w + w0, :].unsqueeze(1)
        vecs = x[:, neigh, :]
        sim = (anc * vecs).sum(-1).mean(dim=1)
        sims.append(sim)
    return torch.stack(sims, dim=1).mean()


class AffinityLoss(nn.Module):
    def __init__(self, anchors=64, window=7):
        super().__init__()
        self.anchors = anchors
        self.window = window

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t_prof = _local_affinity_profile(t.detach(), self.anchors, self.window)
        s_prof = _local_affinity_profile(s, self.anchors, self.window)
        return torch.abs(s_prof - t_prof)


