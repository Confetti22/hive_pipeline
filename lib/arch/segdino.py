import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch, torch.nn as nn

from transformers import AutoModel
from torchvision.transforms import GaussianBlur


import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

def _pca_numpy(X: np.ndarray, k: int = 3) -> np.ndarray:
    """Reduce features to top-k principal components using NumPy SVD.

    Args:
        X: Array of shape (M, C), M = number of tokens/samples, C = channels/features.
        k: Number of principal components to keep.

    Returns:
        (M, k) array: projection of X onto the top-k PCs.
    """
    # center
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # SVD (Xc = U S Vt); PCs are rows of Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Project onto top-k PCs
    return Xc @ Vt[:k].T  # (M, k)

# Wrapper to use DINOv3 models from HuggingFace
class Dinov3HFBackbone(nn.Module):
    """
    Wrap a HuggingFace DINOv3 ViT (e.g., ViT-S/16) so it works as a DPT backbone.

    Exposes:
      - embed_dim: channel dim of tokens (e.g., 384 for ViT-S/16).
      - get_intermediate_layers(x, n): returns list of tensors [B, HW, C] (CLS removed)
        for the transformer block indices given in `n`. `x` is [B,3,H,W] already normalized.
    """
    def __init__(self, model: AutoModel, patch_size: int = 16):
        super().__init__()
        self.model = model.eval()
        self.patch_size = patch_size
        # HF DINOv3 ViTs report hidden size here:
        self.embed_dim = int(model.config.hidden_size)

    @torch.no_grad()
    def get_intermediate_layers(self, x: torch.Tensor, n):
        """
        Args:
          x: [B, 3, H, W] tensor (already preprocessed / normalized to what the HF model expects).
          n: sequence of layer indices, e.g., [2,5,8,11] (0-based over the 12 transformer blocks).

        Returns:
          list of Tensors of length len(n); each is [B, HW, C] (CLS removed).
          H', W' are inferred by your DPT using H//16 and W//16.
        """
        # Run with all hidden states
        out = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)

        # HF returns hidden_states where:
        #   hidden_states[0]  -> after patch/embed (pre-encoder)
        #   hidden_states[1:] -> after block 0..(L-1) respectively
        hs = out.hidden_states
        feats = []
        for idx in n:
            tokens = hs[idx + 1]          # [B, 5+HW, C]
            patch = tokens[:, 5:, :]      # drop CLS + 4REGISTER -> [B, HW, C]
            feats.append(patch)
        return feats

    # (Optional) if you ever call backbone directly; not needed by your DPT
    def forward(self, x: torch.Tensor):
        out = self.model(pixel_values=x, output_hidden_states=False, return_dict=True)
        return out.last_hidden_state  # [B, 1+HW, C]
    



def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch

class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024],
    ):
        super(DPTHead, self).__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        self.scratch.stem_transpose = None
        self.scratch.output_conv = nn.Conv2d(features*4, nclass, kernel_size=1, stride=1, padding=0)  
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        target_hw = layer_1_rn.shape[-2:]  
        layer_2_up = F.interpolate(layer_2_rn, size=target_hw, mode="bilinear", align_corners=True)
        layer_3_up = F.interpolate(layer_3_rn, size=target_hw, mode="bilinear", align_corners=True)
        layer_4_up = F.interpolate(layer_4_rn, size=target_hw, mode="bilinear", align_corners=True)
        fused = torch.cat([layer_1_rn, layer_2_up, layer_3_up, layer_4_up], dim=1)
        out = self.scratch.output_conv(fused)
        return out

class DPT(nn.Module):
    def __init__(
        self, 
        encoder_size='base', 
        nclass=2,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
        backbone = None
    ):
        super(DPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
        }
        
        self.encoder_size = encoder_size
        self.backbone = backbone
        self.feature_map = None
        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)
        
    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        features = self.backbone.get_intermediate_layers(
            x, n = self.intermediate_layer_idx[self.encoder_size]
        )
        #extract the feature for visualization
        if not self.training:
            self.feature_map = self.compute_feature_map_pca(features, patch_h, patch_w)

        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)
        return out

    def compute_feature_map(self,features,patch_h, patch_w):
        "concated feature from 4 layer and upsample to patch_h, patch_w"
        "features: B,C,N  or B,C,N"

        out = []
        for i, x in enumerate(features):
            x = x.permute(0, 2, 1).contiguous().reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).clone().detach()
            out.append(x)
        
        
        
        layer_1, layer_2, layer_3, layer_4 = out
        fused = torch.cat([layer_1, layer_2, layer_3, layer_4], dim=1) #B,C,H,W

        #TODO: maybe need a suitable blur method at feature map to blur out feature variation across cell, 
        # but preserve the difference at region boundary
        # blur = GaussianBlur (kernel_size=3, sigma=0.5)
        # fused = blur(fused)

        up_fused= F.interpolate(fused, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners= False)
        

        up_fused = up_fused.cpu().numpy() # (B,C,H,W)
        up_fused = np.moveaxis(up_fused,1,-1) #(B,H,W,C)
        # remove trivial B dim
        up_fused = np.squeeze(up_fused)
        return up_fused
    

    def compute_feature_map_pca(
        self,
        features: List[torch.Tensor],
        patch_h: int,
        patch_w: int,
        pcs_per_layer: int = 12 
    ) -> np.ndarray:
        """PCA each layer's (M, C) to top-k, concat along channel, reshape, upsample.

        Workflow:
        1) For each tensor x in features, ensure shape is (M, C) where M = B*patch_h*patch_w.
        2) Move x -> CPU NumPy and run PCA to keep `pcs_per_layer` PCs => (M, k).
        3) Concatenate reduced layers along feature dim => (M, k * L).
        4) Reshape to (B, patch_h, patch_w, merged_c).
        5) Upsample spatial dims to (16*patch_h, 16*patch_w) with bilinear interpolation.
        6) Return NumPy array of shape (B, 16*patch_h, 16*patch_w, merged_c).

        Args:
            features: List of tensors; each is either:
                    - (B, HW, C) with HW == patch_h * patch_w, or
                    - (M, C) with M == B * patch_h * patch_w.
            patch_h: Patch height (tokens grid H).
            patch_w: Patch width  (tokens grid W).
            pcs_per_layer: Number of principal components to keep for each layer (default 3).

        Returns:
            NumPy array of shape (B, 16*patch_h, 16*patch_w, merged_c),
            where merged_c = pcs_per_layer * len(features).
        """
        assert len(features) > 0, "features list is empty"

        # Figure out batch size B from the first feature
        x0 = features[0]
        B = int(x0.shape[0])
        M_expected = B * patch_h * patch_w

        reduced_list = []
        for x in features:
            # Normalize shape to (M, C)
            if x.dim() == 3:
                Bx, HW, C = x.shape
                assert Bx == B and HW == patch_h * patch_w, "Feature shape mismatches patch grid"
                x2d = x.reshape(B * patch_h * patch_w, C)
            else:
                M, C = x.shape
                assert M == M_expected, "M does not match B*patch_h*patch_w"
                x2d = x

            # Move to CPU NumPy (as requested) and PCA to top-k
            x_np = x2d.detach().to("cpu", non_blocking=True).float().numpy()
            x_pca = _pca_numpy(x_np, k=pcs_per_layer)  # (M, k)
            reduced_list.append(x_pca)

        # Concatenate along feature/channel dimension: (M, k*L)
        merged = np.concatenate(reduced_list, axis=1)
        merged_c = merged.shape[1]  # k * num_layers

        # Reshape to (B, H, W, C)
        merged = merged.reshape(B, patch_h, patch_w, merged_c)


        t = torch.from_numpy(merged).permute(0, 3, 1, 2)  # (B, C, H, W)

        #smooth the features to avg out the feature variance on cell texture
        blur = GaussianBlur (kernel_size=3, sigma=1)
        t = blur(t)

        up = F.interpolate(
            t,
            size=(patch_h * 16, patch_w * 16),
            mode="bilinear",
            align_corners=True,
        )  # (B, C, 16*H, 16*W)
        up = up.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # (B, 16*H, 16*W, C)

        return up


    def get_feature_map(self):
        if not self.training:
            return self.feature_map
        else:
            return None
