
import torch.nn as nn
from .ae import ConvMLP 
import numpy as np
import torch
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 12),
            nn.ReLU(),
            nn.Linear(12, num_classes)  # Multiclass logits
        )

    def forward(self, x):
        return self.classifier(x)
    

class ConvSegHead(nn.Module):
    """
    A small 3D convolutional head for voxel-wise classification.
    adding padding to insure the output shape is the same as input
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv3d(in_channels // 2, num_classes, kernel_size=1, padding=0)
        )

    def forward(self, x):
        # x shape: [B, C, D, H, W]
        return self.net(x)  # logits per class

class SimpleSegmodel(nn.Module):
    def __init__(self, encoder: nn.Module, seg_head: nn.Module):
        super().__init__()
        self.cmpsd_encoder = encoder
        self.seg_head = seg_head
        self.feature_map = None  # avoid attribute errors when accessed after eval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 3D case -> [B, C, D, H, W]; 2D case -> [B, C, H, W]
        Returns:
            logits tensor shaped like seg_head output
        """
        features = self.cmpsd_encoder(x)


        out = self.seg_head(features)
        out = F.interpolate(out,tuple(x.shape[2:]),mode='trilinear',align_corners=False)

        # If you only want to cache feature maps during eval:
        if not self.training and hasattr(self, "compute_feature_map"):
            # NOTE: adjust this to whatever shape your compute_feature_map expects
            self.feature_map = self.compute_feature_map(features, x.shape[2:])

        return out

    
    def compute_feature_map(self,features,spatial_shape):
        up = F.interpolate(features,tuple(spatial_shape),mode='trilinear', align_corners=False)
        up= up.squeeze(0).cpu().numpy() # [C,D,H,W]
        up = np.moveaxis(up ,0,-1)  #[D,H,W,C]
        return up
    
    
    def get_feature_map(self):
        if not self.training:
            return self.feature_map
        else:
            return None




