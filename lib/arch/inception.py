
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Sequence, Tuple
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
from torchvision.transforms import GaussianBlur

class InceptionBackbone(nn.Module):
    """InceptionV3 feature extractor that returns multi-scale feature maps."""

    def __init__(self, weights: Optional[Inception_V3_Weights] = Inception_V3_Weights.IMAGENET1K_V1):
        super().__init__()
        use_aux = weights is not None
        base = models.inception_v3(weights=weights, aux_logits=use_aux, transform_input=False)
        self.conv1 = base.Conv2d_1a_3x3
        self.conv2 = base.Conv2d_2a_3x3
        self.conv3 = base.Conv2d_2b_3x3
        self.conv4 = base.Conv2d_3b_1x1
        self.conv5 = base.Conv2d_4a_3x3
        self.block5 = nn.Sequential(base.Mixed_5b, base.Mixed_5c, base.Mixed_5d)
        self.block6 = nn.Sequential(base.Mixed_6a, base.Mixed_6b, base.Mixed_6c, base.Mixed_6d, base.Mixed_6e)
        self.block7 = nn.Sequential(base.Mixed_7a, base.Mixed_7b, base.Mixed_7c)
        self.out_channels = [192, 288, 768, 2048]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv4(x)
        x = self.conv5(x)
        feats.append(x)  # 71x71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.block5(x)
        feats.append(x)  # 35x35
        x = self.block6(x)
        feats.append(x)  # 17x17
        x = self.block7(x)
        feats.append(x)  # 8x8
        return feats


class InceptionSegHead(nn.Module):
    """Lightweight multi-scale fusion head similar in spirit to the DPT head."""

    def __init__(self, in_channels: Sequence[int], n_classes: int, proj_dim: int = 128, fuse_dim: int = 128):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, proj_dim, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
            )
            for ch in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(proj_dim * len(in_channels), fuse_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(fuse_dim, n_classes, kernel_size=1)

    def forward(self, feats: Sequence[torch.Tensor], return_fused: bool = False):
        target_hw = feats[0].shape[-2:]
        processed: List[torch.Tensor] = []
        for feat, proj in zip(feats, self.projects):
            x = proj(feat)
            if x.shape[-2:] != target_hw:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            processed.append(x)

        fused = torch.cat(processed, dim=1)
        fused = self.fuse(fused)
        logits = self.classifier(fused)

        if return_fused:
            return logits, fused
        return logits


class InceptionSegModel(nn.Module):
    def __init__(self, backbone: InceptionBackbone, head: InceptionSegHead):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.feature_map: Optional[np.ndarray] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if self.training:
            logits = self.head(feats, return_fused=False)
            fused = None
        else:
            logits, fused = self.head(feats, return_fused=True)

        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if (not self.training) and fused is not None:
            self.feature_map = self._to_feature_map(fused, x.shape[-2:])

        return logits

    def _to_feature_map(self, fused: torch.Tensor, spatial_shape: Tuple[int, int]) -> np.ndarray:
        print(f"in inception_v3: {fused.shape= }, {spatial_shape= }")
        #in inception_v3: fused.shape= torch.Size([1, 128, 92, 92]) B*C*H*W, spatial_shape= torch.Size([384, 384])
        blur = GaussianBlur(kernel_size=11,sigma=4) 
        fused = blur(fused)

        up = F.interpolate(fused, size=spatial_shape, mode="bilinear", align_corners=False)
        
        up = up.detach().cpu().permute(0, 2, 3, 1).contiguous().numpy()
        return np.squeeze(up)

    def get_feature_map(self):
        if not self.training:
            return self.feature_map
        return None
