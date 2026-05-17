import numpy as np
import torch
import torch.nn.functional as F
import zarr
import os
import math
import tempfile
import itertools
from typing import Tuple, Optional, List, Any
from skimage.restoration import denoise_tv_chambolle

from lib.arch.segmodel import Modelsegmodel
from lib.utils.preprocess_img import _ensure_tensor_chw_or_cdhw

"""
Improvements Summary:
1.Itertools.product: Replaced nested for loops with a product generator. This allows the same code to handle 2D and 3D gracefully without separate branches.
2.Slicing with Tuples: Instead of manual y0:y1, x0:x1, we construct slices = tuple(slice(...)) dynamically. This is the "pythonic" way to handle N-dimensional volumes.
3.Encapsulated Logic: The logic for padding, weight calculation, and model forwarding are now independent. If you change your model backbone, you only need to update _model_forward.
4.Feature Mapping: Normalized the feature accumulation to match the probability accumulation logic, ensuring that high-overlap areas don't have "brighter" features.
5.Readability: The eval_full_roi function is now half the length and clearly outlines the steps: Setup -> Loop -> Accumulate -> Normalize.
6.Large Volume Support: Uses Zarr on disk for accumulation buffers when input is a string path.
"""

def get_blend_weight(shape: Tuple[int, ...], cosine_decay: bool = False) -> np.ndarray:
    """Generates N-dimensional blending weights."""
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing='ij')
    dist = np.sqrt(sum(g**2 for g in grid))
    if dist.max() > 0:
        dist /= dist.max()
    if cosine_decay:
        weight = 0.5 * (1.0 + np.cos(dist * np.pi)) 
        return np.where(dist <= 1.0, weight, 0.0).astype(np.float32)
    else:
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

def _ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / b))

def _as_spatial_feature_map(f_map: np.ndarray, dims: int) -> np.ndarray:
    """Return feature map as (*spatial, C), preserving 3D DPT depth batches."""
    f_map = np.asarray(f_map)
    if f_map.ndim == dims + 2 and f_map.shape[0] == 1:
        f_map = f_map[0]
    if f_map.ndim != dims + 1:
        raise ValueError(f"Expected feature map with {dims + 1} dims, got {f_map.shape}")
    return f_map

def _infer_feature_factors(tile: Tuple[int, ...], feature_spatial: Tuple[int, ...]) -> Tuple[int, ...]:
    factors = []
    for tile_size, feat_size in zip(tile, feature_spatial):
        if feat_size <= 0:
            raise ValueError(f"Invalid feature spatial shape: {feature_spatial}")
        factor = max(1, int(round(tile_size / feat_size)))
        if _ceil_div(tile_size, factor) != feat_size:
            raise ValueError(
                f"Feature shape {feature_spatial} is not compatible with tile {tile}; "
                "use tile sizes divisible by the feature stride."
            )
        factors.append(factor)
    return tuple(factors)

def _downsample_weight(weight: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Average an N-D tile weight map into a lower-resolution feature grid."""
    out = np.asarray(weight, dtype=np.float32)
    for axis, target in enumerate(target_shape):
        if out.shape[axis] == target:
            continue
        edges = np.linspace(0, out.shape[axis], target + 1)
        chunks = []
        for i in range(target):
            start = int(math.floor(edges[i]))
            stop = int(math.ceil(edges[i + 1]))
            stop = max(stop, start + 1)
            chunks.append(out.take(indices=range(start, min(stop, out.shape[axis])), axis=axis).mean(axis=axis))
        out = np.stack(chunks, axis=axis)
    return out.astype(np.float32, copy=False)

# -------------------------------------------------------------
#  Inference Core
# -------------------------------------------------------------

def _model_forward(segmodel: Modelsegmodel, tile_img: np.ndarray, device: str) -> Optional[torch.Tensor]:
    """Handles tensor conversion and architecture-specific forward passes."""
    x = _ensure_tensor_chw_or_cdhw(tile_img, segmodel.dims, model_name=segmodel.name).to(device)
    
    # Handle DPT 3D -> 2D slice processing
    if segmodel.name == "DPT" and segmodel.dims == 3 and x.dim() == 5:
        B, C, D, H, W = x.shape
        x2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        logits2d = segmodel.seg_model(x2d)
        f_map = segmodel.seg_model.get_feature_map()
        if f_map is not None and D == 1 and np.asarray(f_map).ndim == 3:
            segmodel.seg_model.feature_map = f_map[None, ...]
        if logits2d is None:
            return None
        logits = logits2d.reshape(B, D, -1, H, W).permute(0, 2, 1, 3, 4)  # B,C,D,H,W
    else:
        logits = segmodel.seg_model(x) #B,C,H,W
        
    return logits

def run_inference_on_tile(
    segmodel: Modelsegmodel,
    tile_img: np.ndarray,
    device: str,
    tv_weight: float,
    collect_prediction: bool = True,
) -> Optional[np.ndarray]:
    """
    Runs model and applies softmax + optional TV denoise.
    """
    logits = _model_forward(segmodel, tile_img, device)
    if not collect_prediction:
        return None
    if logits is None:
        raise RuntimeError("Model did not return logits while collect_prediction=True.")
    
    # Softmax on channel dimension (1 for BCDHW/ BCHW)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    out = np.empty_like(probs)

    if tv_weight > 0:
        for b in range(probs.shape[0]):
            out[b] = denoise_tv_chambolle(probs[b], weight=tv_weight, channel_axis=0)
    
    return out if tv_weight > 0 else probs  # B,C,D,H,W or B,C,H,W

# -------------------------------------------------------------
#  Tiling Logic
# -------------------------------------------------------------

def _tiled_inference_loop(
    segmodel: Modelsegmodel,
    image: Any,
    prob_acc: Any,
    weight_acc: Any,
    feat_acc: Any,
    feat_weight_acc: Any,
    tile: Tuple[int, ...],
    img_shape: Tuple[int, ...],
    feat_shape: Optional[Tuple[int, ...]],
    feat_factors: Optional[Tuple[int, ...]],
    steps: List[int],
    blend: np.ndarray,
    device: str,
    tv_denoise_weight: float,
    capture_features: bool,
    collect_prediction: bool,
) -> Tuple[Any, Any]:
    """Core loop for tiled inference, supporting both numpy and Zarr accumulators."""
    dims = segmodel.dims
    iter_axes = [range(0, img_shape[i], steps[i]) for i in range(dims)]
    
    with torch.no_grad():
        for top_left in itertools.product(*iter_axes):
            # Define slice for input
            slices = tuple(slice(start, min(start + t, img_shape[i])) 
                          for i, (start, t) in enumerate(zip(top_left, tile)))
            
            # Extract and pad tile
            tile_img = np.array(image[slices]) # Ensure numpy for processing
            tile_img = pad_tile(tile_img, tile)
            
            # Inference
            actual_shape = tuple(s.stop - s.start for s in slices)
            extract_slice = tuple(slice(0, s) for s in actual_shape)
            w = blend[extract_slice]

            probs_tile = run_inference_on_tile(
                segmodel,
                tile_img,
                device,
                tv_denoise_weight,
                collect_prediction=collect_prediction,
            )

            if collect_prediction:
                probs_tile = np.squeeze(probs_tile, axis=0)  # remove batch dim
                prob_tile_weighted = probs_tile[(slice(None),) + extract_slice] * w
                prob_acc[(slice(None),) + slices] += prob_tile_weighted
                weight_acc[slices] += w
            
            # Feature Accumulation
            if capture_features:
                f_map = segmodel.seg_model.get_feature_map() # [spatial, C] or [Batch, spatial, C]
                if f_map is not None:
                    f_map = _as_spatial_feature_map(f_map, dims)
                    if feat_factors is None:
                        feat_factors = _infer_feature_factors(tile, f_map.shape[:dims])
                    if feat_shape is None:
                        feat_shape = tuple(_ceil_div(s, f) for s, f in zip(img_shape, feat_factors))
                    feat_slices = tuple(
                        slice(s.start // factor, _ceil_div(s.stop, factor))
                        for s, factor in zip(slices, feat_factors)
                    )
                    feat_actual_shape = tuple(s.stop - s.start for s in feat_slices)
                    feat_extract_slice = tuple(slice(0, s) for s in feat_actual_shape)
                    feat_w = _downsample_weight(w, feat_actual_shape)

                    if feat_acc is None and isinstance(image, np.ndarray):
                        feat_acc = np.zeros((*feat_shape, f_map.shape[-1]), dtype=np.float32)
                        feat_weight_acc = np.zeros(feat_shape, dtype=np.float32)

                    if feat_acc is not None:
                        feat_acc[feat_slices + (slice(None),)] += (
                            f_map[feat_extract_slice + (slice(None),)] * feat_w[..., None]
                        )
                    if feat_weight_acc is not None:
                        feat_weight_acc[feat_slices] += feat_w
    
    return feat_acc, feat_weight_acc

def eval_full_roi(
    segmodel: Modelsegmodel,
    image: Any, # np.ndarray or str path
    device: str = "cuda",
    tile: Optional[Tuple[int, ...]] = None,
    capture_features: bool = True,
    tv_denoise_weight: float = 1.0,
    overlap: float = 0.25,
    zarr_temp_dir: Optional[str] = None,
    collect_prediction: bool = True,
    feature_output_path: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[Any]]:
    
    dims = segmodel.dims
    
    # 1. Handle Input Source
    if isinstance(image, str):
        # Open as Zarr or using helper
        try:
            image_data = zarr.open(image, mode='r')
            img_shape = image_data.shape[:dims]
        except:
            from helper.image_reader import wrap_image
            image_data = wrap_image(image)
            # Adapt to image_reader objects
            if hasattr(image_data, 'info'):
                img_shape = tuple(image_data.info[0]['data_shape'])
            elif hasattr(image_data, 'image'):
                img_shape = image_data.image.shape[:dims]
            else:
                img_shape = image_data.shape[:dims]
        is_large = True
    else:
        image_data = image
        img_shape = image.shape[:dims]
        is_large = feature_output_path is not None or not isinstance(image, np.ndarray)

    # Adjust tile for 3D if needed
    if tile is not None:
        tile = tile if dims == 3 else tile[1:]

    # 2. Fallback for non-tiled inference
    if tile is None or all(t >= i for t, i in zip(tile, img_shape)):
        with torch.no_grad():
            input_img = np.array(image_data) if is_large else image_data
            probs = run_inference_on_tile(
                segmodel,
                input_img,
                device,
                tv_denoise_weight,
                collect_prediction=collect_prediction,
            )
            feat = segmodel.seg_model.get_feature_map() if capture_features else None
            if not collect_prediction:
                return None, feat
            probs = np.squeeze(probs, axis=0)  # remove batch dim
            return np.argmax(probs, axis=0) + 1, feat

    # 3. Setup Accumulators
    temp_dir = None
    feat_weight_acc = None
    feat_shape = None
    feat_factors = None
    if is_large:
        if zarr_temp_dir is None:
            temp_dir = tempfile.mkdtemp()
            zarr_temp_dir = temp_dir
        
        # Determine chunk sizes for Zarr
        chunks = tuple(min(s, 128) for s in img_shape)

        if collect_prediction:
            prob_acc = zarr.open(os.path.join(zarr_temp_dir, 'prob_acc.zarr'), mode='w', 
                                shape=(segmodel.n_classes, *img_shape), dtype=np.float32, 
                                chunks=(1, *chunks))
            weight_acc = zarr.open(os.path.join(zarr_temp_dir, 'weight_acc.zarr'), mode='w', 
                                  shape=img_shape, dtype=np.float32, chunks=chunks)
        else:
            prob_acc = None
            weight_acc = None
        
        feat_acc = None
    else:
        if collect_prediction:
            prob_acc = np.zeros((segmodel.n_classes, *img_shape), dtype=np.float32)
            weight_acc = np.zeros(img_shape, dtype=np.float32)
        else:
            prob_acc = None
            weight_acc = None
        feat_acc = None
    
    blend = get_blend_weight(tile)
    steps = [max(1, int(t * (1 - overlap))) for t in tile]
    
    # 4. Run Tiled Inference Loop
    # If Zarr feat_acc is needed, we might need a preliminary step to get feat_dim
    if is_large and capture_features:
        # Get feature dim from a small tile
        dummy_slices = tuple(slice(0, min(t, s)) for t, s in zip(tile, img_shape))
        dummy_tile = np.array(image_data[dummy_slices])
        dummy_tile = pad_tile(dummy_tile, tile)
        with torch.no_grad():
            _ = run_inference_on_tile(
                segmodel,
                dummy_tile,
                device,
                0,
                collect_prediction=collect_prediction,
            )
            f_map = segmodel.seg_model.get_feature_map()
            if f_map is not None:
                f_map = _as_spatial_feature_map(f_map, dims)
                feat_dim = f_map.shape[-1]
                feat_factors = _infer_feature_factors(tile, f_map.shape[:dims])
                feat_shape = tuple(_ceil_div(s, f) for s, f in zip(img_shape, feat_factors))
                chunks_feat = tuple(min(s, 128) for s in feat_shape) + (feat_dim,)
                chunks_weight = tuple(min(s, 128) for s in feat_shape)
                feat_store_path = feature_output_path or os.path.join(zarr_temp_dir, 'feat_acc.zarr')
                feat_weight_path = os.path.join(zarr_temp_dir, 'feat_weight_acc.zarr')
                feat_acc = zarr.open(feat_store_path, mode='w', 
                                    shape=(*feat_shape, feat_dim), dtype=np.float32, chunks=chunks_feat)
                feat_weight_acc = zarr.open(feat_weight_path, mode='w',
                                           shape=feat_shape, dtype=np.float32, chunks=chunks_weight)

    feat_acc, feat_weight_acc = _tiled_inference_loop(
        segmodel, image_data, prob_acc, weight_acc, feat_acc, feat_weight_acc,
        tile, img_shape, feat_shape, feat_factors, steps, blend, device,
        tv_denoise_weight, capture_features, collect_prediction
    )

    # 5. Normalization
    if is_large:
        pred = None
        if collect_prediction:
            # Normalize in blocks for efficiency if it's Zarr
            for i in range(img_shape[0]):
                w = weight_acc[i]
                w_norm = np.maximum(w, 1e-8)
                prob_acc[:, i] /= w_norm

            # Pred also needs to be argmaxed in blocks
            pred = np.zeros(img_shape, dtype=np.uint8)
            for i in range(img_shape[0]):
                pred[i] = np.argmax(prob_acc[:, i], axis=0) + 1

        if feat_acc is not None and feat_weight_acc is not None:
            for i in range(feat_acc.shape[0]):
                w = np.maximum(feat_weight_acc[i], 1e-8)
                feat_acc[i] /= w[..., None]
    else:
        pred = None
        if collect_prediction:
            prob_acc /= np.maximum(weight_acc, 1e-8)
            pred = np.argmax(prob_acc, axis=0) + 1
        if feat_acc is not None and feat_weight_acc is not None:
            feat_acc /= np.maximum(feat_weight_acc[..., None], 1e-8)
        
    return pred, feat_acc
