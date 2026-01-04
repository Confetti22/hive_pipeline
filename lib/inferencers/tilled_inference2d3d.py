import numpy as np
from lib.arch.segmodel import Modelsegmodel
from lib.utils.preprocess_img import _ensure_tensor_chw_or_cdhw
import torch
import torch.nn.functional as F
from skimage.restoration import denoise_tv_chambolle
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from skimage.restoration import denoise_tv_chambolle
from typing import Tuple, Optional, List
from lib.arch.segmodel import Modelsegmodel
from lib.utils.preprocess_img import _ensure_tensor_chw_or_cdhw

"""
Improvements Summary:
1.Itertools.product: Replaced nested for loops with a product generator. This allows the same code to handle 2D and 3D gracefully without separate branches.
2.Slicing with Tuples: Instead of manual y0:y1, x0:x1, we construct slices = tuple(slice(...)) dynamically. This is the "pythonic" way to handle N-dimensional volumes.
3.Encapsulated Logic: The logic for padding, weight calculation, and model forwarding are now independent. If you change your model backbone, you only need to update _model_forward.
4.Feature Mapping: Normalized the feature accumulation to match the probability accumulation logic, ensuring that high-overlap areas don't have "brighter" features.
5.Readability: The eval_full_roi function is now half the length and clearly outlines the steps: Setup -> Loop -> Accumulate -> Normalize.
"""
def get_blend_weight(shape: Tuple[int, ...]) -> np.ndarray:
    """Generates N-dimensional cosine-like blending weights."""
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing='ij')
    dist = np.sqrt(sum(g**2 for g in grid))
    if dist.max() > 0:
        dist /= dist.max()
    return np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)

def pad_tile(tile_img: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Pads tile to target_shape using reflect or edge mode."""
    curr_shape = tile_img.shape[:len(target_shape)]
    if curr_shape == target_shape:
        return tile_img
    
    padding = []
    for curr, target in zip(curr_shape, target_shape):
        padding.append((0, target - curr))
    
    # Add padding for channel dim if it exists
    if tile_img.ndim > len(target_shape):
        padding.append((0, 0))
        
    mode = "edge" if any(s == 1 for s in curr_shape) else "reflect"
    return np.pad(tile_img, tuple(padding), mode=mode)

# -------------------------------------------------------------
#  Inference Core
# -------------------------------------------------------------

def _model_forward(segmodel: Modelsegmodel, tile_img: np.ndarray, device: str) -> torch.Tensor:
    """Handles tensor conversion and architecture-specific forward passes."""
    x = _ensure_tensor_chw_or_cdhw(tile_img, segmodel.dims, model_name=segmodel.name).to(device)
    
    # Handle DPT 3D -> 2D slice processing
    if segmodel.name == "DPT" and segmodel.dims == 3 and x.dim() == 5:
        B, C, D, H, W = x.shape
        x2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        logits2d = segmodel.seg_model(x2d)
        logits = logits2d.reshape(B, D, -1, H, W).permute(0, 2, 1, 3, 4)
    else:
        logits = segmodel.seg_model(x)
        
    return logits

def run_inference_on_tile(segmodel, tile_img, device, tv_weight):
    """Runs model and applies softmax + optional TV denoise."""
    logits = _model_forward(segmodel, tile_img, device)
    
    # Softmax on channel dimension (usually 1 for BCDHW, 0 if squeezed)
    probs = F.softmax(logits, dim=1 if logits.dim() > 3 else 0).detach().cpu().numpy()
    probs = np.squeeze(probs, axis=0) # Remove Batch
    
    if tv_weight > 0:
        probs = denoise_tv_chambolle(probs, weight=tv_weight, channel_axis=0)
    
    return probs

# -------------------------------------------------------------
#  Tiling Logic
# -------------------------------------------------------------

def eval_full_roi(
    segmodel: Modelsegmodel,
    image: np.ndarray,
    device: str = "cuda",
    tile: Optional[Tuple[int, ...]] = None,
    capture_features: bool = True,
    tv_denoise_weight: float = 1.0,
    overlap: float = 0.25,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    
    dims = segmodel.dims
    img_shape = image.shape[:dims]
    
    # 1. Fallback for non-tiled inference
    if tile is None or all(t >= i for t, i in zip(tile, img_shape)):
        with torch.no_grad():
            probs = run_inference_on_tile(segmodel, image, device, tv_denoise_weight)
            feat = segmodel.seg_model.get_feature_map() if capture_features else None
            return np.argmax(probs, axis=0) + 1, feat

    # 2. Setup Accumulators
    prob_acc = np.zeros((segmodel.n_classes, *img_shape), dtype=np.float32)
    weight_acc = np.zeros(img_shape, dtype=np.float32)
    
    feat_acc = None
    blend = get_blend_weight(tile)
    
    # Compute step sizes based on overlap
    steps = [max(1, int(t * (1 - overlap))) for t in tile]
    
    # 3. Sliding Window
    # Create ranges for N-dimensions
    iter_axes = [range(0, img_shape[i], steps[i]) for i in range(dims)]
    import itertools

    with torch.no_grad():
        for top_left in itertools.product(*iter_axes):
            # Define slice for input
            slices = tuple(slice(start, min(start + t, img_shape[i])) 
                          for i, (start, t) in enumerate(zip(top_left, tile)))
            
            # Extract and pad tile
            tile_img = pad_tile(image[slices], tile)
            
            # Inference
            probs_tile = run_inference_on_tile(segmodel, tile_img, device, tv_denoise_weight)
            
            # Probability Accumulation
            # Crop probs back to the actual size (in case it was padded)
            actual_shape = tuple(s.stop - s.start for s in slices)
            extract_slice = tuple(slice(0, s) for s in actual_shape)
            
            w = blend[extract_slice]
            prob_acc[(slice(None),) + slices] += probs_tile[(slice(None),) + extract_slice] * w
            weight_acc[slices] += w
            
            # Feature Accumulation
            if capture_features:
                f_map = segmodel.seg_model.get_feature_map() # [spatial, C] or [Batch, spatial, C]
                if f_map is not None:
                    if f_map.ndim > dims + 1: f_map = f_map[0] # remove batch dim
                    
                    if feat_acc is None:
                        feat_acc = np.zeros((*img_shape, f_map.shape[-1]), dtype=np.float32)
                    
                    feat_acc[slices + (slice(None),)] += f_map[extract_slice + (slice(None),)] * w[..., None]

    # 4. Normalization
    prob_acc /= np.maximum(weight_acc, 1e-8)
    pred = np.argmax(prob_acc, axis=0) + 1
    
    if feat_acc is not None:
        feat_acc /= np.maximum(weight_acc[..., None], 1e-8)
        
    return pred, feat_acc
