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
from typing import List, Tuple, Optional

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
      - embed_dim: channel dim of tokens (e.g., 768 for ViT-S/16).
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
    scratch.layer_rns = nn.ModuleList([
        nn.Conv2d(in_ch, out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        for in_ch in in_shape
    ])
    return scratch

class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=128, 
        use_bn=False, 
        out_channels=[256, 512, 1024],
    ):
        super(DPTHead, self).__init__()
        # in_channels  is  the embed_dim of backbone is 768
        # out_channels  = [96, 192, 384, 768]
        self.num_features = len(out_channels)
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        # out_channels  = [96, 192, 384, 768]
        # features =128 
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        self.scratch.stem_transpose = None
        self.scratch.output_conv = nn.Conv2d(features*4, nclass, kernel_size=1, stride=1, padding=0)  

    def forward(self, out_features, patch_h,patch_w):
        if len(out_features) != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {len(out_features)}")

        processed = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.scratch.layer_rns[i](x)
            processed.append(x)

        target_hw = processed[0].shape[-2:]
        upsampled = [processed[0]]
        for feat in processed[1:]:
            upsampled.append(F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=True))
        fused = torch.cat(upsampled, dim=1)
        return self.scratch.output_conv(fused)


class DPTHead_warped(nn.Module):
        def __init__(
            self, 
            nclass,
            in_channels, 
            features=128, 
            use_bn=False, 
            out_channels=[256, 512, 1024],
            patch_h = None,
            patch_w = None,
            ):  
            super(DPTHead_warped, self).__init__()
            self.patch_h = patch_h
            self.patch_w = patch_w
            self.head = DPTHead(nclass, in_channels, features, use_bn, out_channels)

        def forward(self, out_features, scale_factor=16):
            out = self.head(out_features, self.patch_h, self.patch_w)
            B,C,H,W = out.shape
            # blur = GaussianBlur(kernel_size=3, sigma=1)
            # features = blur(features)
            out = F.interpolate(out, (out.shape[-2]*scale_factor, out.shape[-1]*scale_factor), mode='bilinear', align_corners=True)
    
            return out


class DPT(nn.Module):
    def __init__(
        self, 
        encoder_size='base', 
        nclass=2,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
        backbone = None,
        seg_head_layers = [2,5,8,11],
        feat_up_method: str = "bilinear",
        smooth_params=(16,4,1),
    ):
        super(DPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
        }
        
        self.encoder_size = encoder_size
        self.backbone = backbone
        self.feature_map = None
        self.feat_up_method = feat_up_method
        self.smooth_params = smooth_params

        default_layers = self.intermediate_layer_idx.get(self.encoder_size, self.intermediate_layer_idx['base'])
        self.seg_head_layers = list(seg_head_layers) if seg_head_layers is not None else default_layers

        base_out_channels = out_channels
        if isinstance(base_out_channels, int):
            base_out_channels = [base_out_channels]
        else:
            base_out_channels = list(base_out_channels)

        if len(base_out_channels) == 1:
            seg_out_channels = base_out_channels * len(self.seg_head_layers)
        elif len(base_out_channels) == len(self.seg_head_layers):
            seg_out_channels = base_out_channels
        elif len(base_out_channels) == len(default_layers):
            layer_to_channels = {idx: ch for idx, ch in zip(default_layers, base_out_channels)}
            seg_out_channels = [layer_to_channels.get(idx, base_out_channels[-1]) for idx in self.seg_head_layers]
        else:
            raise ValueError(
                f"out_channels must be length 1, {len(self.seg_head_layers)}, or {len(default_layers)} "
                f"but got {len(base_out_channels)}"
            )

        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=seg_out_channels)
        

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        features = self.backbone.get_intermediate_layers(
            x, 
            n = self.intermediate_layer_idx[self.encoder_size],
        )
        #extract the feature for visualization in eval mode
        if not self.training:
            self.feature_map = self.compute_feature_map_pca(features, patch_h, patch_w)

        out = self.head(features, patch_h, patch_w)
         
        out = self.upsample_feature_map(out, scale_factor=16)

        return out #logits
    
    def upsample_feature_map(self, features, scale_factor):
        """
        upsample feature map both for inference and feature_map computation
        """
        if self.feat_up_method == "bilateral" and (not self.training):
            print(f"traing{self.training} upsample feature map using bilateral upsample with kernel_size={self.smooth_params[0]}, spatial_sigma={self.smooth_params[1]}, range_sigma={self.smooth_params[2]}")
            out  = self._bilateral_upsample_feature_map(
                features,
                target_hw=(features.shape[-2]*scale_factor, features.shape[-1]*scale_factor),
                kernel_size=self.smooth_params[0],
                spatial_sigma=self.smooth_params[1],
                range_sigma=self.smooth_params[2],
                chunk_size=4,
            )
        else:
            print(f"training{self.training} upsample feature map using bilinear upsample")
            blur = GaussianBlur(kernel_size=3, sigma=1)
            features = blur(features)
            out = F.interpolate(features, (features.shape[-2]*scale_factor, features.shape[-1]*scale_factor), mode='bilinear', align_corners=True)
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

        up_fused = self.upsample_feature_map(fused,scale_factor=16)
        

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
        pcs_per_layer: List[int] = [24,120,120,24],
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

            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            x_np = x2d.detach().to("cpu", non_blocking=True).float().numpy()
            scaler = StandardScaler()
            x_std = scaler.fit_transform(x_np)
            if pcs_per_layer is not None:
                k = pcs_per_layer[len(reduced_list)]
                pca = PCA(n_components=k)
                x_pca = pca.fit_transform(x_std)
            else:
                pca = PCA(n_components=0.85, svd_solver='full')
                x_pca = pca.fit_transform(x_std)
                print(f"{pca.n_components_}_components  chosen to explain 85% variance: ")
    
            reduced_list.append(x_pca)

        # Concatenate along feature/channel dimension: (M, k*L)
        merged = np.concatenate(reduced_list, axis=1)
        merged_c = merged.shape[1]  # k * num_layers

        # Reshape to (B, H, W, C)
        merged = merged.reshape(B, patch_h, patch_w, merged_c)


        t = torch.from_numpy(merged).permute(0, 3, 1, 2)  # (B, C, H, W)

        up = self.upsample_feature_map(t, scale_factor=16)  # (B, C, 16*H, 16*W)

        up = up.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # (B, 16*H, 16*W, C)
        # remove trivial B dim
        up = np.squeeze(up)
        if up.mean() == 0.0:
            print("feature map all zero!")

        return up

    @torch.no_grad()
    def _bilateral_upsample_feature_map(
        self,
        t: torch.Tensor,
        target_hw: Tuple[int, int],
        kernel_size: int = 16,
        spatial_sigma: float = 8,
        range_sigma: float = 1,
        guide_channels: int = 0,
        chunk_size: int = 16,
    ) -> torch.Tensor:
        """Edge-preserving upsample using a bilateral filter on the feature map."""
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if kernel_size % 2 == 0:
            kernel_size += 1

        up = F.interpolate(
            t,
            size=target_hw,
            mode="bilinear",
            align_corners=True,
        )
        if kernel_size == 1:
            return up

        guide = up
        if guide_channels <= 0:
            guide = up.mean(dim=1, keepdim=True)
        else:
            guide = up[:, : min(up.shape[1], guide_channels)]

        guide = guide.float()
        guide_std = float(guide.std().clamp(min=1e-6))
        sigma_range = max(range_sigma * guide_std, 1e-6)
        sigma_spatial = max(spatial_sigma, 1e-6)

        pad = kernel_size // 2
        coords = torch.arange(kernel_size, device=up.device, dtype=guide.dtype) - pad
        grid_y, grid_x = torch.meshgrid(coords, coords)
        spatial_w = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2.0 * sigma_spatial ** 2))
        spatial_w = spatial_w.reshape(1, -1, 1)

        guide_pad = F.pad(guide, (pad, pad, pad, pad), mode="reflect")
        guide_unfold = F.unfold(guide_pad, kernel_size=kernel_size)
        B, _, hw = guide_unfold.shape
        g = guide.shape[1]
        guide_unfold = guide_unfold.view(B, g, kernel_size * kernel_size, hw)
        guide_center = guide.reshape(B, g, hw).unsqueeze(2)
        diff = guide_unfold - guide_center
        range_w = torch.exp(-(diff * diff).sum(dim=1) / (2.0 * sigma_range ** 2))
        weights = range_w * spatial_w
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        up_pad = F.pad(up, (pad, pad, pad, pad), mode="reflect")
        B, C, H, W = up.shape
        out = torch.empty_like(up)
        if chunk_size <= 0:
            chunk_size = C
        chunk_size = min(chunk_size, C)

        for start in range(0, C, chunk_size):
            end = min(C, start + chunk_size)
            up_chunk = up_pad[:, start:end, :, :]
            up_unfold = F.unfold(up_chunk, kernel_size=kernel_size)
            up_unfold = up_unfold.view(B, end - start, kernel_size * kernel_size, hw)
            out_chunk = (up_unfold * weights.unsqueeze(1)).sum(dim=2)
            out[:, start:end, :, :] = out_chunk.view(B, end - start, H, W)

        if out.dtype != t.dtype:
            out = out.to(dtype=t.dtype)
        return out


    def get_feature_map(self):
        if not self.training:
            return self.feature_map
        else:
            return None
class LinearTokenSeg(nn.Module):
    """
    Minimal segmentation head: a single linear projection on the final DINOv3 tokens.
    """
    def __init__(self, backbone, nclass=8, encoder_size='base'):
        super().__init__()
        self.backbone = backbone
        self.encoder_size = encoder_size
        self.classifier = nn.Linear(self.backbone.embed_dim, nclass)
        self.layer_idx = {
            'small': [11],
            'base': [11],
        }
        self.patch_size = getattr(self.backbone, 'patch_size', 16)

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        features = self.backbone.get_intermediate_layers(
            x, n=self.layer_idx.get(self.encoder_size, self.layer_idx['base'])
        )
        tokens = features[-1]  # [B, HW, C]
        logits = self.classifier(tokens)  # [B, HW, nclass]
        B = logits.shape[0]
        logits = logits.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)
        logits = F.interpolate(
            logits, (patch_h * self.patch_size, patch_w * self.patch_size), mode='bilinear', align_corners=True
        )
        return logits

