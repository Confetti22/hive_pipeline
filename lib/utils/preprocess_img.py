from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
import numpy as np
import torch
from skimage import exposure


def pad_to_multiple_of_unit(img,unit = 8):
    H, W = img.shape[-2:]  # assuming shape [..., H, W]

    pad_H = (unit - H % unit) % unit
    pad_W = (unit - W % unit) % unit

    pad_top = 0
    pad_bottom = pad_H
    pad_left = 0
    pad_right = pad_W

    img_padded = np.pad(
        img,
        pad_width=[(0, 0)] * (img.ndim - 2) + [(pad_top, pad_bottom), (pad_left, pad_right)],
        mode='reflect'  # or 'constant' for zero padding
    )
    return img_padded

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def preprocess_uint16_for_imagenet(
    img: np.ndarray,
    *,
    make_3ch: bool = True,
    robust_percentiles: Optional[Tuple[float, float]] = (1.0, 99.9),
    log_transform: bool = False,
    gamma: Optional[float] = None,
    clamp01_after_rescale: bool = True,
) -> torch.Tensor:
    """Convert a uint16 fluorescence image to ImageNet-normalized tensor.
    
    Args:
        img: Input image. Accepts 2D (H, W), 3D (D,H, W), dtype can be uint16/uint8/float.
        make_3ch: If True and input is single-channel, replicate to 3 channels for ImageNet backbones.
        robust_percentiles: Percentiles for robust rescaling (lo, hi). Set to None to disable.
        log_transform: If True, apply log1p before rescaling (useful for heavy-tailed fluorescence).
        gamma: Optional gamma correction (e.g., 0.8–1.2). Applied after rescale.
        clamp01_after_rescale: Clamp to [0,1] at the end of intensity remapping.
    
    Returns:
        torch.Tensor: Tensor of shape (C, D, H, W) with D=1. If make_3ch=True → C=3, else C=1 (if input was 1ch).
                      ImageNet-normalized (z-standardized) float32 tensor.
    
    Logic:
        1) Convert to float32 and roughly scale into [0,1] (dtype-aware).
        2) Optional log1p to compress dynamic range.
        3) Optional robust percentile rescale to improve contrast and reduce outlier influence.
        4) Optional gamma.
        5) Channel handling (repeat single channel to 3 if requested).
        6) ImageNet z-standardization using mean/std in [0,1] range.
        7) Return tensor in (C, D, H, W) with D=1 to fit a unified loader.
    """

    # ---- to float in [0,1] (initial pass)
    arr = img
    arr = np.asarray(arr)
    arr = arr.astype(np.float32)
    if img.dtype == np.uint16:
        arr = arr / 65535.0
    elif img.dtype == np.uint8:
        arr = arr / 255.0
    else:
        # Assume already float-like; gently bring into [0,1] if dynamic range is large
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
        if vmax > 1.0 or vmin < 0.0:
            rng = max(vmax - vmin, 1e-12)
            arr = (arr - vmin) / rng

    # ---- optional log to tame bright spikes
    if log_transform:
        arr = np.log1p(arr * 1000.0)  # scale before log to increase separation of dim structures
        # re-normalize to [0,1]
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-12)

    # ---- optional robust rescale by percentiles
    if robust_percentiles is not None:
        lo, hi = robust_percentiles
        lo_v, hi_v = np.percentile(arr, [lo, hi])
        if hi_v > lo_v:
            arr = exposure.rescale_intensity(arr, in_range=(lo_v, hi_v), out_range=(0.0, 1.0))

    # ---- optional gamma
    if gamma is not None and gamma > 0:
        arr = np.power(np.clip(arr, 0.0, 1.0), gamma)

    if clamp01_after_rescale:
        arr = np.clip(arr, 0.0, 1.0)

    # ---- channel handling
    if make_3ch:
        arr = np.repeat(arr[..., None], 3, axis=-1)  # (H,W,3) or (D,H,W,3)
    else:
        arr = arr[..., None]  # (H,W,1) or (D,H,W,1)

    # ---- to torch (C,H,W) or (c,d,h,w)
    arr = np.moveaxis(arr,-1,0)
    t = torch.from_numpy(arr).contiguous().to(torch.float32)

    # ---- ImageNet z-standardization (expect inputs in [0,1])
    if t.shape[0]== 3:
        if len(t.shape) ==3:
            t = (t - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        else:
            t = (t - IMAGENET_MEAN[:, None, None,None]) / IMAGENET_STD[:, None, None,None]

    elif t.shape[0] == 1:
        # If staying grayscale, you can either:
        #  (A) standardize with grayscale stats (mean=0.5,std=0.5) OR
        #  (B) convert first conv to one channel (see helper below)
        # Here we keep simple grayscale z-norm:
        gray_mean, gray_std = 0.5, 0.5
        t = (t - gray_mean) / gray_std
    else:
        raise ValueError(f"Unexpected channel count: {t.shape[0]}")

    # ---- return as (C, D, H, W) with D=1 to match your loader template
    if len(t.shape) ==3:
        return t.unsqueeze(1)
    else:
        return t

import numpy as np
import torch


def preprocess_uint8rgb_for_imagenet(img: np.ndarray) -> torch.Tensor:
    """
    Preprocess an RGB uint8 image (H,W,3) or (D,H,W,3)for DINO model input.

    Steps:
    1. Rescale to [0,1]
    2. Transpose to channel-first (C,H,W)
    3. Normalize per-channel
    4. Add batch and dummy depth channel -> shape [B, C, D, H, W]

    Args:
        img (np.ndarray): Input image (H, W, 3), dtype=uint8, values [0,255].

    Returns:
        torch.Tensor: Preprocessed tensor with shape [1, 3, 1, H, W].
    """

    # 1. rescale
    arr = img.astype(np.float32) / 255.0   # [D,H,W,3] [H,W,3] -> float32 in [0,1]

    # 2. transpose
    arr = np.moveaxis(arr,-1,0) #[3,H,W]  [3,D,H,W]

    # 3. normalize
    if len(arr.shape) ==3:
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
        std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    else:
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None,None]
        std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None,None]

    arr = (arr - mean) / std               # per-channel normalize

    # 4. add depth channel
    tensor = torch.from_numpy(arr)
    if len(arr.shape ) == 3:
        tensor = tensor.unsqueeze(1)  # [3,1,H,W]
    return tensor
