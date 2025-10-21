from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import to_rgb_for_vit
from .teacher import TeacherDinoV3
from .student import TinyViTWithTaps, tokens_from_cnn_bottleneck
from .losses import FeatureMimicCosine, AffinityLoss


class Distiller(nn.Module):
    def __init__(self,
                 teacher_dir: str,
                 student_type: str = "cnn",
                 student_cnn_builder=None,
                 student_tinyvit_name: str = "vit_tiny_patch16_224",
                 taps_teacher_1based=(2, 6, 11),
                 taps_student_1based=(2, 6, 11),
                 lambda_feat: float = 1.0,
                 lambda_aff: float = 0.25,
                 use_spatial_coords: bool = False):
        super().__init__()
        self.teacher = TeacherDinoV3(teacher_dir)
        self.taps_t = [k - 1 for k in taps_teacher_1based]
        self.student_type = student_type
        self.lambda_feat = lambda_feat
        self.lambda_aff = lambda_aff
        self.use_spatial_coords = use_spatial_coords

        if student_type == "cnn":
            assert student_cnn_builder is not None, "Provide student_cnn_builder(args)->model"
            self.student = student_cnn_builder()
            self.feat_mode = "vit2cnn"
        else:
            self.student = TinyViTWithTaps(student_tinyvit_name, pretrained=False)
            self.student.register_taps([k - 1 for k in taps_student_1based])
            self.feat_mode = "vit2vit"

        self.adapter: Optional[nn.Module] = None
        self.coord_proj: Optional[nn.Linear] = None
        self.feat_loss = FeatureMimicCosine(mode=self.feat_mode)
        self.aff_loss = AffinityLoss(anchors=64, window=7)
        self.cache: Dict[str, List[torch.Tensor]] = {}

    def _maybe_build_adapter(self, student_dim: int):
        tdim = self.teacher.embed_dim
        if self.adapter is None:
            self.adapter = nn.Identity() if student_dim == tdim else nn.Linear(student_dim, tdim, bias=False)

    def _maybe_build_coord_proj(self):
        if self.coord_proj is None:
            self.coord_proj = nn.Linear(2, self.teacher.embed_dim, bias=False)

    @staticmethod
    def _add_token_coords(toks: torch.Tensor) -> torch.Tensor:
        b, n, _ = toks.shape
        h = w = int(math.sqrt(n)); assert h * w == n
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=toks.device),
            torch.linspace(-1, 1, w, device=toks.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1).reshape(1, n, 2).repeat(b, 1, 1)
        return coords

    def forward_teacher_tokens(self, x_rgb: torch.Tensor, image_ids: List[str]) -> List[torch.Tensor]:
        b = x_rgb.size(0)
        per_sample: List[List[torch.Tensor]] = []
        for i in range(b):
            key = image_ids[i]
            if key in self.cache:
                toks = self.cache[key]
            else:
                toks = self.teacher.forward_tokens(x_rgb[i:i+1], self.taps_t)
                toks = [t.squeeze(0) for t in toks]
                self.cache[key] = toks
            per_sample.append([t.unsqueeze(0) for t in toks])
        k = len(self.taps_t)
        tapwise: List[List[torch.Tensor]] = [[] for _ in range(k)]
        for bi in range(b):
            for ki in range(k):
                tapwise[ki].append(per_sample[bi][ki])
        tapwise = [torch.cat(tlist, dim=0) for tlist in tapwise]
        return tapwise

    def forward_student_tokens(self, x_in: torch.Tensor) -> List[torch.Tensor]:
        if self.student_type == "cnn":
            bottleneck, _ = self.student(x_in)
            b, cs, h, w = bottleneck.shape
            self._maybe_build_adapter(cs)
            toks = tokens_from_cnn_bottleneck(bottleneck)
            toks = self.adapter(toks)
            if self.use_spatial_coords:
                self._maybe_build_coord_proj()
                coord_feats = self._add_token_coords(toks)
                toks = toks + self.coord_proj(coord_feats)
            return [toks for _ in self.taps_t]
        else:
            toks_list = self.student.forward_tokens(x_in, [k for k in self.taps_t])
            self._maybe_build_adapter(self.student.embed_dim)
            toks_list = [self.adapter(t) for t in toks_list]
            if self.use_spatial_coords:
                self._maybe_build_coord_proj()
                out_list: List[torch.Tensor] = []
                for t in toks_list:
                    coord_feats = self._add_token_coords(t)
                    out_list.append(t + self.coord_proj(coord_feats))
                toks_list = out_list
            return toks_list

    def compute_losses(self, s_list: List[torch.Tensor], t_list: List[torch.Tensor]):
        k = len(s_list)
        l_feat = 0.0
        l_aff = 0.0
        for s, t in zip(s_list, t_list):
            l_feat = l_feat + self.feat_loss(s, t)
            l_aff = l_aff + self.aff_loss(s, t)
        l_feat /= k
        l_aff /= k
        total = self.lambda_feat * l_feat + self.lambda_aff * l_aff
        return total, l_feat.detach(), l_aff.detach()

    def forward(self, x_gray: torch.Tensor, image_ids: List[str]) -> Dict[str, torch.Tensor]:
        x_rgb = to_rgb_for_vit(x_gray)
        t_list = self.forward_teacher_tokens(x_rgb, image_ids)
        s_list = self.forward_student_tokens(x_gray)
        loss, lfeat, laff = self.compute_losses(s_list, t_list)
        return {"loss": loss, "L_feat": lfeat, "L_aff": laff}


