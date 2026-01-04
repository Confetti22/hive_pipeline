#!/usr/bin/env python3
"""
Interactive Segmentation UI (v3)
================================
Single-viewer Napari workflow that:
  • Loads a 2D or 3D ROI image
  • Creates a zero-initialized labels layer for sparse user annotations
  • Lets the user choose a segmentation model architecture (cmpsd, DPT, or inception_v3)
  • Freezes the backbone and trains only a lightweight seghead from sparse labels
  • Evaluates full-ROI prediction with tiling when needed
  • Captures a feature volume (pre-seghead) for similarity tools
  • Supports dims={2,3} pipelines and per-slice eval for DPT on 3D
  • Preserves double-click: 1D similarity plots (X/Y at current Z) + 3D NCC map

Design notes
-----------
- No dual-viewers and no SimpleViewer widgets. Only one viewer.
- No precomputed featuremap; features are captured from the active backbone.
- Aux/registered masks are intentionally NOT loaded.
- The number of seghead output channels is derived from user label integers.
- DPT uses a DINOv3 backbone + DPTHead (segdino.py). cmpsd uses conv+mlp + ConvSegHead (seg.py). inception_v3 freezes a pretrained Inception backbone and fuses multi-scale features with a lightweight head.

Dependencies
------------
- napari, magicgui, numpy, torch, torchvision (for DINO/DPT utils), scikit-image, zarr (optional)
- Your project modules: lib.arch.seg(ConvSegHead), segdino.DPTHead, lib.arch.ae/build_encoders, helper.one_dim_statis (optional)

"""


from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, Optional, Sequence, Tuple
from collections import defaultdict

import napari
from time import time
from magicgui import magicgui
from qtpy.QtWidgets import QMessageBox
 
from skimage.restoration import denoise_tv_chambolle
from helper.napari_view_utilis import _filter_layer_name_with_pattern
from lib.utils.preprocess_img import  preprocess_uint16_for_imagenet, preprocess_uint8rgb_for_imagenet, pad_to_multiple
from lib.datasets.sparse_label_dataset import SparseLabelSegDataset

from confettii.plot_helper import three_pca_as_rgb_image
from confettii.grah_cut import n_cut
# ----------------------------------
# Project bootstrap (repo root import)
# ----------------------------------
DOWNSAMPLE = True 
NAPARI = True

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)



def _round_and_clip_bbox(y0: float, x0: float, y1: float, x1: float, H: int, W: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert float bbox coords to int pixel indices, clipped to [0,H/W], and ensure non-empty.
    Returns (y0, y1, x0, x1) in inclusive-exclusive indexing, or None if invalid.
    """
    y_min = max(0, int(np.floor(min(y0, y1))))
    y_max = min(H, int(np.ceil(max(y0, y1))))
    x_min = max(0, int(np.floor(min(x0, x1))))
    x_max = min(W, int(np.ceil(max(x0, x1))))

    if y_max <= y_min or x_max <= x_min:
        return None
    return (y_min, y_max, x_min, x_max)

def find_valid_rectangle_bbox_from_shapes(state: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    """
    Inspect state['layers']['shapes'] (a napari Shapes layer) and extract a valid rectangle bbox
    in pixel coords relative to the current ROI.
    
    Rules:
      - Uses the last rectangle in the layer that is visible.
      - Accepts napari rectangle data given as 4 corner vertices [[y,x], ...].
      - Validates bbox lies within the current ROI spatial size.

    Returns:
      (y0, y1, x0, x1) in int, inclusive-exclusive indexing; or None if no valid rectangle.

    Assumes:
      - state['roi'] is an array whose first two dims are (H, W) (RGB or grayscale both fine).
      - state['layers']['shapes'] is a napari shapes layer or None.
    """
    
    layers = state.get("layers", {})

    shape_layer_name_set = _filter_layer_name_with_pattern(layers,name_patterns=['Shapes'])
    if len(shape_layer_name_set) == 0:
        return None
    shape_layer_name = next(iter(shape_layer_name_set))

    verts = layers[shape_layer_name].data


    roi = state['roi']
    if roi is None:
        return None
    dims = state['dims']
    if dims ==2:
        H,W = roi.shape[:dims]
    else:
        D,H,W = roi.shape[:dims]
    
    verts = np.asarray(verts)  # expected shape (4, 2): [[y,x], ...]
    verts = np.squeeze(verts)
    ys = verts[:, -2]
    xs = verts[:, -1]
    bbox = _round_and_clip_bbox(ys.min(), xs.min(), ys.max(), xs.max(), H, W)
    if bbox is not None:
        return bbox

    return None


def _uses_imagenet_preproc(model_name: str) -> bool:
    """Return True if the model expects ImageNet-style 3-channel normalized input."""
    return model_name in {'s_tinyvit', 's_tinyvittimm', "DPT", "inception_v3", "inception_v3_single", "inception_v3_preavg_single"}


def _ensure_tensor_chw_or_cdhw(img: np.ndarray, dims: int,model_name:str) -> torch.Tensor:
    """Convert numpy image to torch tensor with shape (B=1, C=1, H, W) or (B=1, C=1, D, H, W).

    Args:
        img: Input image with shape (H,W) or (D,H,W).
        dims: 2 or 3.
    Returns:
        Torch tensor ready for model input.
    """
    imagenet_preproc = _uses_imagenet_preproc(model_name)

    if dims == 2:  

        if imagenet_preproc:              #DPT branch 
            if len(img.shape)==2:
                t = preprocess_uint16_for_imagenet(img)
            else:
                t = preprocess_uint8rgb_for_imagenet(img)


            if t.shape[1] ==1: #DPT model avoid D 
                t= t.squeeze(1)

            t = t.unsqueeze(0) #add B

        else:                             #other model 
            t = torch.from_numpy(img.astype(np.float32))[None, None] #B*C*H*W
    else:
        if imagenet_preproc:
            if len(img.shape) ==3:
                t = preprocess_uint16_for_imagenet(img)
            else:
                t = preprocess_uint8rgb_for_imagenet(img)

            # Keep the depth axis (even if size==1) so downstream 3D tiling logic stays consistent.
            t = t.unsqueeze(0) #add B

        else:
            t = torch.from_numpy(img.astype(np.float32))[None,None] #B*C*D*H*W

    return t

# ----------------------------------
# Model registry & builders
# ----------------------------------

# ----------------------------------
# Training/evaluation helpers
# ----------------------------------


from lib.arch.segmodel import Modelsegmodel
def train_seghead(segmodel: Modelsegmodel,
                  dataset: Dataset,
                  n_classes: int,
                  device: str = "cuda",
                  epochs: int = 5,
                  batch_size: int = 64,
                  lr: float = 1e-3) -> None:
    """Train only the seghead with the backbone frozen.

    Args:
        segmodel: Modelsegmodel (backbone frozen)
        dataset: training dataset from sparse labels
        n_classes: number of classes
        device: 'cuda' or 'cpu'
        epochs: small number for interactivity
        batch_size: mini-batch size
        lr: learning rate
    """
    
    # Freeze backbone should be done in model initialization
    segmodel.seg_model.to(device)

    #drop_last False to ensure nonempty loader when  len(ds) ==1 (this is true when input img and train_roi is the same) 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last= False)
    opt = torch.optim.AdamW(segmodel.seg_model.parameters(), lr=lr)
    
    #~~~~~~~ weighted l1 loss ~~~~~~~~#
    from lib.loss.ce_dice_combo import ComboLoss
    from lib.utils.loss_utils import compute_class_weights_from_dataset

    class_weights = compute_class_weights_from_dataset(dataset, num_classes=n_classes,recon_target_flag=False)
    loss_fn = ComboLoss(class_weights=class_weights, focal=True)

    # before training loop (once)
    torch.autograd.set_detect_anomaly(True)
    for n_epoch in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y = y.squeeze(1) 
            if x.shape[2] ==1:
                x = x.squeeze(2)
            if segmodel.name == "DPT" and x.dim() == 5:
                # slice 3D into 2D batches externally
                B, C, D, H, W = x.shape
                x2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
                logits2d = segmodel.seg_model(x2d)  # [B*D, C', H, W]
                logits = logits2d.reshape(B, D, n_classes, H, W).permute(0, 2, 1, 3, 4)
            else:
                logits = segmodel.seg_model(x) #[B,C,D,H,W] or [B,C,H,W]

            # Align logits & labels to same spatial dims
            if logits.dim() == y.dim() + 1:# 2D: logits [B,C,H,W], y [B,H,W] OK# 3D: logits [B,C,D,H,W], y [B,D,H,W] OK
                pass
            else:
                raise RuntimeError("Unexpected logits/labels dims mismatch")
            
            logits_flat = logits.permute(0,*range(2, logits.ndim),1)[ y>= 0]  # [N, K]
            labels_flat = y[ y>= 0]

            # inside the loop, right before backward
            logits_flat = logits_flat.contiguous()
            assert logits_flat.dtype in (torch.float32, torch.float16, torch.bfloat16)
            assert logits_flat.requires_grad, "logits_flat must require grad"
            assert labels_flat.dtype == torch.long, f"labels dtype is {labels_flat.dtype}, expected long"
            assert labels_flat.min().item() >= 0 and labels_flat.max().item() < n_classes, \
                f"label out of range: [{labels_flat.min().item()}, {labels_flat.max().item()}]"
            
            total_loss, ce_loss, dice_loss= loss_fn(logits_flat, labels_flat.long())

            opt.zero_grad(set_to_none=True)
            
            total_loss.backward()  # if it fails, detect_anomaly will print the offending op
            opt.step()

            print(f"epoch:{n_epoch}:train_loss: {total_loss.item():.4f}  "
                    f"(ce={ce_loss.item():.4f}, dice={dice_loss.item():.4f})")




# -------------------------------------------------------------
#  Blend weight generators
# -------------------------------------------------------------
def _make_blend_weight_2d(h, w):
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    dist = np.sqrt(x * x + y * y)
    dist /= dist.max() if dist.max() > 0 else 1.0
    weight = 1.0 - dist
    weight = np.clip(weight, 0.0, 1.0)
    return weight.astype(np.float32)


def _make_blend_weight_3d(d: int, h: int, w: int) -> np.ndarray:
    z = np.linspace(-1, 1, d)[:, None, None]
    y = np.linspace(-1, 1, h)[None, :, None]
    x = np.linspace(-1, 1, w)[None, None, :]
    dist = np.sqrt(z * z + y * y + x * x)
    if dist.max() > 0:
        dist /= dist.max()
    weight = 1.0 - dist
    return weight.astype(np.float32)

def _run_single_2d(segmodel:Modelsegmodel, image, device, capture_features, tv_weight):
    """
    Run full-image 2D inference, exactly reproducing the logic from your
    original eval_full_roi() 2D no-tile branch.
    """
    x = _ensure_tensor_chw_or_cdhw(image, dims=2, model_name=segmodel.name).to(device)

    logits = segmodel.seg_model(x).squeeze(0)  # [C,H,W]
    probs = F.softmax(logits, dim=0).cpu().numpy()

    # optional TV denoise
    # if tv_weight > 0:
    #     probs = denoise_tv_chambolle(probs, weight=tv_weight, channel_axis=0)

    pred = np.argmax(probs, axis=0) + 1

    # features before segmentation head
    fvol = segmodel.seg_model.get_feature_map() if capture_features else None

    return pred, fvol


# -------------------------------------------------------------
#  Single full-image inference (no tiling) for 2D
# -------------------------------------------------------------
def _run_single_3d(segmodel, image, device, capture_features, tv_weight):
    """
    Run full-image 3D inference, preserving DPT slicing and original logic.
    """
    dims = 3
    n_classes = segmodel.n_classes

    x = _ensure_tensor_chw_or_cdhw(image, dims=3, model_name=segmodel.name).to(device)

    # ---- DPT special 3D→2D path ----
    if segmodel.name == "DPT" and x.dim() == 5:
        B, C, D, H, W = x.shape
        x2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        logits2d = segmodel.seg_model(x2d)  # [BD, C', H, W]
        logits = logits2d.reshape(B, D, n_classes, H, W).permute(0, 2, 1, 3, 4)
    else:
        logits = segmodel.seg_model(x)  # [B,C,D,H,W] or [1,C,D,H,W]

    probs = F.softmax(logits, dim=1).cpu().numpy()  # [B,C,D,H,W]

    if tv_weight > 0:
        probs = denoise_tv_chambolle(probs, weight=tv_weight, channel_axis=1)

    pred = np.argmax(probs, axis=1) + 1
    pred = np.squeeze(pred)  # remove batch dim

    fvol = segmodel.seg_model.get_feature_map() if capture_features else None

    return pred, fvol


def eval_full_roi(
    segmodel: Modelsegmodel,
    image: np.ndarray,
    device: str = "cuda",
    tile: Optional[Tuple[int, ...]] = None,
    capture_features: bool = True,
    tv_denoise_weight: float = 100000,
    overlap: float = 0.25,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    dims = segmodel.dims
    n_classes = segmodel.n_classes

    print(f"eval with tv_denoise weight {tv_denoise_weight}, overlap {overlap}")

    with torch.no_grad():
        # --------------------
        # 2D BRANCH
        # --------------------
        if dims == 2:
            H, W = image.shape[:2]
            if tile is None:
                raise ValueError("tile must be provided for overlapped tiling")
            ph, pw = tile

            # no tiling (full-image) fallback
            if ph >= H and pw >= W:
                return _run_single_2d(segmodel, image, device, capture_features, tv_denoise_weight)

            oh = int(ph * overlap)
            ow = int(pw * overlap)

            prob_acc = np.zeros((n_classes, H, W), dtype=np.float32)
            weight_acc = np.zeros((H, W), dtype=np.float32)
            fvol = None
            fweight = None

            blend = _make_blend_weight_2d(ph, pw)  # (ph, pw)

            for y0 in range(0, H, max(1, ph - oh)):
                for x0 in range(0, W, max(1, pw - ow)):
                    y1 = min(H, y0 + ph)
                    x1 = min(W, x0 + pw)

                    tile_img = image[y0:y1, x0:x1]
                    
                    #pad the tile image if it is smaller than the tile size
                    th_pad = max(0, ph - tile_img.shape[0])
                    tw_pad = max(0, pw - tile_img.shape[1])
                    if th_pad > 0 or tw_pad > 0:
                        if tile_img.ndim == 2:
                            pad_spec = ((0, th_pad), (0, tw_pad))
                        elif tile_img.ndim == 3:
                            pad_spec = ((0, th_pad), (0, tw_pad), (0, 0))
                        else:
                            raise ValueError(f"Unexpected tile ndim {tile_img.ndim}")
                        pad_mode = "reflect"
                        if (tile_img.shape[0] == 1 and th_pad > 0) or (tile_img.shape[1] == 1 and tw_pad > 0):
                            pad_mode = "edge"
                        tile_img = np.pad(tile_img, pad_spec, mode=pad_mode)
                    x = _ensure_tensor_chw_or_cdhw(tile_img, 2, model_name=segmodel.name).to(device)

                    # logits: [C,h,w]
                    logits = segmodel.seg_model(x).squeeze(0)
                    probs = F.softmax(logits, dim=0).detach().cpu().numpy()  # [C,h,w]

                    # probs spatial size
                    th, tw = probs.shape[1:]

                    # weight crop for this tile
                    w_h = min(th, blend.shape[0], y1 - y0)
                    w_w = min(tw, blend.shape[1], x1 - x0)
                    weight = blend[:w_h, :w_w]  # [w_h, w_w]

                    # optional TV denoise
                    if tv_denoise_weight > 0:
                        probs_d = denoise_tv_chambolle(probs[:, :w_h, :w_w],
                                                       weight=tv_denoise_weight,
                                                       channel_axis=0)
                    else:
                        probs_d = probs[:, :w_h, :w_w]

                    prob_acc[:, y0:y0+w_h, x0:x0+w_w] += probs_d * weight[None, :, :]
                    weight_acc[y0:y0+w_h, x0:x0+w_w] += weight

                    # ---- feature maps with same blending ----
                    if capture_features:
                        f_tile = segmodel.seg_model.get_feature_map()  # could be [H,W,C] or [B,H,W,C]


                        fh, fw, fc = f_tile.shape
                        fh_eff = min(fh, w_h)
                        fw_eff = min(fw, w_w)

                        if fvol is None:
                            fvol = np.zeros((H, W, fc), dtype=np.float32)
                            fweight = np.zeros((H, W), dtype=np.float32)

                        fvol[y0:y0+fh_eff, x0:x0+fw_eff, :] += (
                            f_tile[:fh_eff, :fw_eff, :] * weight[:fh_eff, :fw_eff, None]
                        )
                        fweight[y0:y0+fh_eff, x0:x0+fw_eff] += weight[:fh_eff, :fw_eff]

            # normalize probability volume
            prob_acc /= weight_acc[None, :, :]
            pred = np.argmax(prob_acc, axis=0) + 1  # [H,W]

            # normalize feature volume (if requested)
            if capture_features and fvol is not None:
                fvol /= fweight[..., None]
            else:
                fvol = None

            return pred, fvol

        # --------------------
        # 3D BRANCH
        # --------------------
        else:
            D, H, W = image.shape[:3]
            if tile is None:
                raise ValueError("tile must be provided for overlapped tiling")
            pd, ph, pw = tile

            if pd >= D and ph >= H and pw >= W:
                return _run_single_3d(segmodel, image, device, capture_features, tv_denoise_weight)

            od = int(pd * overlap)
            oh = int(ph * overlap)
            ow = int(pw * overlap)

            step_d = max(1, pd - od)
            step_h = max(1, ph - oh)
            step_w = max(1, pw - ow)

            prob_acc = np.zeros((n_classes, D, H, W), dtype=np.float32)
            weight_acc = np.zeros((D, H, W), dtype=np.float32)
            fvol = None
            fweight = None

            blend = _make_blend_weight_3d(pd, ph, pw)

            for z0 in range(0, D, step_d):
                for y0 in range(0, H, step_h):
                    for x0 in range(0, W, step_w):
                        z1 = min(D, z0 + pd)
                        y1 = min(H, y0 + ph)
                        x1 = min(W, x0 + pw)

                        tile_img = image[z0:z1, y0:y1, x0:x1]

                        dz_pad = max(0, pd - tile_img.shape[0])
                        dy_pad = max(0, ph - tile_img.shape[1])
                        dx_pad = max(0, pw - tile_img.shape[2])
                        if dz_pad > 0 or dy_pad > 0 or dx_pad > 0:
                            if tile_img.ndim == 3:
                                pad_spec = ((0, dz_pad), (0, dy_pad), (0, dx_pad))
                            elif tile_img.ndim == 4:
                                pad_spec = ((0, dz_pad), (0, dy_pad), (0, dx_pad), (0, 0))
                            else:
                                raise ValueError(f"Unexpected tile ndim {tile_img.ndim}")
                            pad_mode = "reflect"
                            if (
                                (tile_img.shape[0] == 1 and dz_pad > 0)
                                or (tile_img.shape[1] == 1 and dy_pad > 0)
                                or (tile_img.shape[2] == 1 and dx_pad > 0)
                            ):
                                pad_mode = "edge"
                            tile_img = np.pad(tile_img, pad_spec, mode=pad_mode)

                        x = _ensure_tensor_chw_or_cdhw(tile_img, 3, model_name=segmodel.name).to(device)

                        if segmodel.name == "DPT" and x.dim() == 5:
                            B, C, Dt, Ht, Wt = x.shape
                            x2d = x.permute(0, 2, 1, 3, 4).reshape(B * Dt, C, Ht, Wt)
                            logits2d = segmodel.seg_model(x2d)
                            logits = logits2d.reshape(B, Dt, n_classes, Ht, Wt).permute(0, 2, 1, 3, 4)
                        else:
                            logits = segmodel.seg_model(x)

                        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                        probs = np.squeeze(probs, axis=0)  # [C, d, h, w]
                        if probs.ndim == 3:  # handle models that dropped the depth axis (D=1)
                            probs = probs[:, None, ...]
                        elif probs.ndim != 4:
                            raise ValueError(f"Unexpected probs shape {probs.shape}, expected 4 dims.")

                        td, th, tw = probs.shape[1:]
                        w_d = min(td, blend.shape[0], z1 - z0)
                        w_h = min(th, blend.shape[1], y1 - y0)
                        w_w = min(tw, blend.shape[2], x1 - x0)
                        weight = blend[:w_d, :w_h, :w_w]

                        prob_slice = probs[:, :w_d, :w_h, :w_w]
                        if tv_denoise_weight > 0:
                            prob_slice = denoise_tv_chambolle(
                                prob_slice, weight=tv_denoise_weight, channel_axis=0
                            )

                        prob_acc[:, z0:z0+w_d, y0:y0+w_h, x0:x0+w_w] += prob_slice * weight[None, :, :, :]
                        weight_acc[z0:z0+w_d, y0:y0+w_h, x0:x0+w_w] += weight

                        if capture_features:
                            f = segmodel.seg_model.get_feature_map()
                            if f is None:
                                continue
                            if f.ndim == 5:
                                f_tile = f[0]
                            elif f.ndim == 4:
                                f_tile = f
                            else:
                                raise ValueError(f"Unexpected feature map shape {f.shape}")

                            fd, fh, fw, fc = f_tile.shape
                            fd_eff = min(fd, w_d)
                            fh_eff = min(fh, w_h)
                            fw_eff = min(fw, w_w)

                            if fvol is None:
                                fvol = np.zeros((D, H, W, fc), dtype=np.float32)
                                fweight = np.zeros((D, H, W), dtype=np.float32)

                            weight_slice = weight[:fd_eff, :fh_eff, :fw_eff]
                            fvol[z0:z0+fd_eff, y0:y0+fh_eff, x0:x0+fw_eff, :] += (
                                f_tile[:fd_eff, :fh_eff, :fw_eff, :] * weight_slice[..., None]
                            )
                            fweight[z0:z0+fd_eff, y0:y0+fh_eff, x0:x0+fw_eff] += weight_slice

            prob_acc /= weight_acc[None, :, :, :]
            pred = np.argmax(prob_acc, axis=0) + 1

            if capture_features and fvol is not None:
                fvol /= fweight[..., None]
            else:
                fvol = None

            return pred, fvol


# Similarity / NCC utilities
# ----------------------------------
def cosine_similarity_map(anchor: np.ndarray, feat: np.ndarray) -> np.ndarray:
    """Cosine similarity between an anchor feature and every location.

    Args:
        anchor: (C,) feature vector at the clicked location.
        feat:   (H, W, C) or (D, H, W, C) channel-last feature map.

    Returns:
        (H, W) map for 2D, (D, H, W) volume for 3D.
    """
    a = anchor.astype(np.float32)
    a_norm = np.linalg.norm(a) + 1e-8

    if feat.ndim == 3:
        H, W, C = feat.shape
        f = feat.reshape(-1, C).astype(np.float32)      # [HW, C]
        num = f @ a                                     # [HW]
        den = (np.linalg.norm(f, axis=1) * a_norm) + 1e-8
        s = num / den
        return s.reshape(H, W)

    elif feat.ndim == 4:
        D, H, W, C = feat.shape
        f = feat.reshape(-1, C).astype(np.float32)      # [DHW, C]
        num = f @ a
        den = (np.linalg.norm(f, axis=1) * a_norm) + 1e-8
        s = num / den
        return s.reshape(D, H, W)

    else:
        raise ValueError(f"feat.ndim must be 3 or 4, got {feat.ndim}")

def ncc3d(anchor: np.ndarray, feat: np.ndarray) -> np.ndarray:
    """Voxelwise normalized cross-correlation for 3D channel-last features.

    Args:
        anchor: (C,) anchor vector.
        feat:   (D, H, W, C) channel-last feature volume.

    Returns:
        (D, H, W) NCC volume in [-1, 1] (approximately).
    """
    if feat.ndim != 4:
        raise ValueError(f"Expected feat of shape (D,H,W,C), got ndim={feat.ndim}")

    D, H, W, C = feat.shape
    a = anchor.astype(np.float32)
    f = feat.reshape(-1, C).astype(np.float32)          # [DHW, C]

    # Z-normalize across channel dimension
    a = (a - a.mean()) / (a.std() + 1e-8)               # [C]
    f_mean = f.mean(axis=1, keepdims=True)              # [DHW, 1]
    f_std  = f.std(axis=1, keepdims=True) + 1e-8
    f = (f - f_mean) / f_std                            # [DHW, C]

    s = (f * a).sum(axis=1) / C                         # [DHW]
    return s.reshape(D, H, W)   


# ----------------------------------
# Napari UI wiring
# ----------------------------------

def _ask(viewer: napari.Viewer, title: str, text: str) -> None:
    m = QMessageBox(viewer.window._qt_window)
    m.setWindowTitle(title)
    m.setText(text)
    m.exec_()


def add_ui(viewer: napari.Viewer) -> None:
    state = {
        "roi": None,              # np.ndarray (H,W) or (D,H,W)
        "labels": None,           # np.ndarray same shape as roi
        "segmodel": None,           # Modelsegmodel
        "pred": None,             # np.ndarray prediction
        "feat": None,             # np.ndarray feature volume [C,H,W] or [C,D,H,W]
        "dims": 3,
    }

    if "roi" in viewer.layers:
        viewer.layers.remove("roi")
    if "user_labels" in viewer.layers:
        viewer.layers.remove("user_labels")

    # roi,label,mask = load_3d_rm009()
    # roi,label,mask = load_DKROI() 
    from lib.datasets.load_rois import load_t1779_3
    roi, label,mask = load_t1779_3()
    roi_shape = roi.shape[:state["dims"]]

    state["roi"] = roi
    state["labels"] = label if label is not None else np.zeros(roi_shape,dtype=np.uint8)
    state["mask"] =  mask if mask is not None else np.ones(roi_shape,dtype=bool)

    viewer.add_image(state["roi"], name="roi")
    viewer.add_labels(state["labels"], name="user_labels")
    viewer.add_labels(state["mask"], name="mask",opacity=0.3)

    state['layers'] = viewer.layers
    



    # --- Build model button ---
    from lib.arch.segmodel import build_cmpsd, build_dpt, build_inception_v3
    @magicgui(call_button="Build Model",
              arch={"choices": ["cmpsd", "DPT", "inception_v3"]})
    def build_model_widget(arch: str = "DPT"):
        """Instantiate backbone+seghead given current labels for channel count."""
        roi = state["roi"]
        labels = state["labels"]
        if roi is None or labels is None:
            _ask(viewer, "Build Model", "Load ROI first and create labels.")
            return
        classes = np.unique(labels)   
        n_classes = max(2, int(len(classes) -  1))  #ignore the unlabled part 0  ensure >= 2
        dims = state["dims"]
        if arch == "cmpsd":
            state["segmodel"] = build_cmpsd(dims=dims, n_classes=n_classes)
        elif arch == "inception_v3":
            state["segmodel"] = build_inception_v3(dims=dims, n_classes=n_classes)
        else:
            state["segmodel"] = build_dpt(dims=dims, n_classes=n_classes)
        _ask(viewer, "Build Model", f"Built {arch} with n_classes={n_classes}, dims={dims}")

    # --- Train seghead ---
   # shared widget config (note: you wrote "tw_", assuming you meant "tv_")
    COMMON_WIDGETS = dict(
        tile_d={"widget_type": "LineEdit"},
        tile_h={"widget_type": "LineEdit"},
        tile_w={"widget_type": "LineEdit"},
        tv_denoise_weight={"widget_type": "LineEdit"},
    )
     


    def train_widget(epochs: int = 20, batch_size: int = 16, lr: float = 1e-4,
                     patch_h: int =1536 , patch_w: int = 1536,
                     patch_d: int = 1):
        """Train lightweight seghead from sparse labels.

        For dims=2, uses (patch_h, patch_w). For dims=3, uses (patch_d, patch_h, patch_w).
        """
        # Force conversion to int
        patch_d = int(patch_d)
        patch_h = int(patch_h)
        patch_w = int(patch_w)
        segmodel: Modelsegmodel = state["segmodel"]
        roi = state["roi"]
        labels = state["labels"]

        if segmodel is None and NAPARI:
            _ask(viewer, "Train", "Build the model first.")
            return
        if (labels is None or np.count_nonzero(labels) == 0) and NAPARI:
            _ask(viewer, "Train", "Please draw some labels on 'user_labels' layer.")
            return

        dims = state["dims"]
        imagenet_preproc = _uses_imagenet_preproc(segmodel.name)

        if dims == 2:
            ds = SparseLabelSegDataset(roi, labels, dims=2, patch_size=(patch_h, patch_w),imagenet_preproc=imagenet_preproc)
        else:
            ds = SparseLabelSegDataset(roi, labels, dims=3, patch_size=(patch_d, patch_h, patch_w),imagenet_preproc=imagenet_preproc)

        n_classes = max(2, len(np.unique(labels))-1)

        train_seghead(segmodel, ds, n_classes=n_classes, device="cuda" if torch.cuda.is_available() else "cpu",
                      epochs=epochs, batch_size=batch_size, lr=lr)
        # _ask(viewer, "Train", "Training finished.")
        print(f"Training finished.")

    
    def _eval_widget(tile_h: int , tile_w: int , tile_d: int,
                     tv_denoise_weight : float,
                     capture_features: bool = False):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        segmodel: Modelsegmodel = state["segmodel"]
        segmodel.seg_model.eval()
        segmodel.seg_model = segmodel.seg_model.to(device)

        roi = state["roi"]

        if segmodel is None or roi is None:
            _ask(viewer, "Evaluate", "Load ROI and build the model first.")
            return

        dims = state["dims"]
        tile = None
        if dims == 2 and (tile_h > 0 and tile_w > 0):
            tile = (tile_h, tile_w)
        elif dims == 3 and (tile_d > 0 and tile_h > 0 and tile_w > 0):
            tile = (tile_d, tile_h, tile_w)
        

        if segmodel.name =="DPT":
            roi = pad_to_multiple(roi, 16,dims=dims)
        
        # Try to discover a valid bbox from the shapes layer
        if NAPARI:
            bbox = find_valid_rectangle_bbox_from_shapes(state)
        else:
            bbox = None

        if bbox is None:
            pred, feat = eval_full_roi(segmodel, roi, device, tile=tile, capture_features=capture_features,tv_denoise_weight=tv_denoise_weight)
            offset = (0,0) if dims ==2 else (0,0,0) #z,y,x
            state['final_roi'] = roi

        else:
            roi_spatial_shape = roi.shape[:dims]
            y0,y1,x0,x1 = bbox
            offset = (y0,x0) if dims ==2 else (0,y0,x0)

            roi_win = roi[y0:y1,x0:x1] if dims==2 else roi[:,y0:y1,x0:x1] 
            padded_roi_win = pad_to_multiple(roi_win,16, dims=dims)

            pred_win, feat_win = eval_full_roi(segmodel, padded_roi_win,device, tile=tile, capture_features=capture_features, tv_denoise_weight=tv_denoise_weight)
            state['final_roi'] = padded_roi_win 

            if dims == 3:
                pred_win = pred_win[:,:y1-y0,:x1-x0] #D,H,W
                feat_win = feat_win[:,:y1-y0,:x1-x0]   if capture_features else None
            else:
                pred_win = pred_win[:y1-y0,:x1-x0] #H,W
                feat_win = feat_win[:,:y1-y0,:x1-x0]   if capture_features else None

            pred,feat= pred_win,feat_win


        # pred[pred==1] = 0 #set backgound(1) to 0 to do not display
        state["pred"], state["feat"] = pred, feat
        state['offset'] = offset 
        print(f"eval done, pred shape: {pred.shape}, feat shape: {feat.shape if feat is not None else None}, offset: {offset}")

        # tsne_plot(feat,state["labels"])


        if "prediction" in viewer.layers and NAPARI:
            viewer.layers.remove("prediction")
        viewer.add_labels(pred, name="prediction",translate=offset)
        # _ask(viewer, "Evaluate", "Prediction layer added. Feature volume captured." if capture_features else "Prediction done.")
        print(f"Prediction layer added. Feature volume captured." if capture_features else "Prediction done.")

        def _ensure_feat_rgb_layer():
            feat = state.get("feat", None)
            translation = state['offset'] 

            
            if feat is None and NAPARI:
                _ask(viewer, "PCA-RGB", "Capture features first (Evaluate with capture_features=True).")
                return

            mask = state.get("mask", None)
            translation = state.get("offset", (0, 0, 0))

            if state["dims"] == 2:
                H, W, C = feat.shape
                spatial_shape = (H, W)
            else:
                D, H, W, C = feat.shape
                spatial_shape = (D, H, W)

            mask_bool = None
            if mask is not None:
                mask_view = np.asarray(mask)
                if mask_view.shape != spatial_shape:
                    try:
                        slices = tuple(
                            slice(int(translation[i]), int(translation[i]) + spatial_shape[i])
                            for i in range(len(spatial_shape))
                        )
                        mask_view = mask_view[slices]
                    except Exception:
                        mask_view = None
                if mask_view is not None and mask_view.shape == spatial_shape:
                    mask_bool = mask_view.astype(bool)

            flat_feat = feat.reshape(-1, C)
            if mask_bool is None:
                rgb = three_pca_as_rgb_image(flat_feat, spatial_shape)
            else:
                mask_flat = mask_bool.reshape(-1)
                rgb_flat = np.zeros((mask_flat.size, 3), dtype=np.float32)
                if mask_flat.any():
                    masked_rgb = three_pca_as_rgb_image(flat_feat[mask_flat], (int(mask_flat.sum()),))
                    rgb_flat[mask_flat] = masked_rgb.reshape(-1, 3)
                rgb = rgb_flat.reshape(*spatial_shape, 3)

            if "feat_rgb" in viewer.layers and NAPARI:
                del viewer.layers["feat_rgb"]

            if NAPARI:
                viewer.add_image(rgb, name="feat_rgb", rgb=True, blending="additive",translate=translation)
            state['pca'] = rgb
        
        def _ncut():
            feat = state.get("feat", None)
            translation = state['offset'] 
            mask = state.get("mask", None)

            if feat is None:
                _ask(viewer, "NCut", "Capture features first (Evaluate with capture_features=True).")
                return

            mask_bool = None
            if mask is not None:
                mask_view = np.asarray(mask)
                spatial_shape = feat.shape[:2] if state["dims"] == 2 else feat.shape[:3]
                if mask_view.shape != spatial_shape:
                    try:
                        slices = tuple(
                            slice(int(translation[i]), int(translation[i]) + spatial_shape[i])
                            for i in range(len(spatial_shape))
                        )
                        mask_view = mask_view[slices]
                    except Exception:
                        mask_view = None
                if mask_view is not None and mask_view.shape == spatial_shape:
                    mask_bool = mask_view.astype(bool)

            if state["dims"] == 2:
                H, W, C = feat.shape
                input_image = state['final_roi'] 
                if mask_bool is not None and mask_bool.any():
                    coords = np.argwhere(mask_bool)
                    min_idx = coords.min(axis=0)
                    max_idx = coords.max(axis=0) + 1
                    region_slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(min_idx, max_idx))
                    mask_crop = mask_bool[region_slices]
                    feat_crop = feat[region_slices + (slice(None),)]
                    feat_crop = np.where(mask_crop[..., None], feat_crop, 0)
                    input_crop = input_image[region_slices]
                    ncut_crop = n_cut(feat_crop, input_crop)
                    ncut_map = np.zeros((H, W), dtype=ncut_crop.dtype)
                    ncut_view = ncut_map[region_slices]
                    ncut_view[mask_crop] = ncut_crop[mask_crop]
                else:
                    ncut_map = n_cut(feat,input_image)

            if "ncut" in viewer.layers:
                del viewer.layers["ncut"]
            viewer.add_image(ncut_map, name="ncut", colormap="viridis", blending="additive",translate=translation)

        # create/update PCA-RGB layer immediately if possible
        _ensure_feat_rgb_layer()
        # _ncut()

    def eval_widget(tile_h: int = 1536, tile_w: int = 1536, tile_d: int = 1,
                   tv_denoise_weight : float = 100000,
                    capture_features: bool = True):
        tile_d = int(tile_d)
        tile_h = int(tile_h)
        tile_w = int(tile_w)
        tv_denoise_weight = float(tv_denoise_weight)
        _eval_widget(tile_h,tile_w,tile_d,tv_denoise_weight,capture_features=capture_features)


    @magicgui(
    call_button="Train SegHead and then eval",
    epochs={"min": 1, "max": 50},
    batch_size={"min": 1, "max": 512},
    lr={"step": 1e-4},
    patch_d={"widget_type": "LineEdit"},
    patch_h={"widget_type": "LineEdit"},
    patch_w={"widget_type": "LineEdit"},
    ** COMMON_WIDGETS
    )
    def train_and_eval_widget(
                    epochs: int = 2, batch_size: int = 16, lr: float = 1e-4,
                    patch_h: int =512 , patch_w: int = 512,patch_d: int = 1,

                    tile_h: int = 512, tile_w: int = 512, tile_d: int = 1,
                    tv_denoise_weight : float = 100000,
                    capture_features: bool = True
                     ):
        begin = time()
        train_widget(epochs,batch_size,lr,patch_h,patch_w,patch_d)
        current = time()
        print(f"Training done, train time{current - begin:.4f}")
        eval_widget(tile_h,tile_w,tile_d,tv_denoise_weight,capture_features=capture_features)
        end = time()
        print(f"Train+Eval done. eval time{end -current:.4f}, total time{end - begin:.4f}")
        

    from lib.arch.segmodel import build_and_load_weights_dpt
    @magicgui(
        call_button="eval SegHead pretrained",
        **COMMON_WIDGETS
    )
    def eval_widget_predefined(tile_h: int = 512, tile_w: int = 512, tile_d: int = 1,
                   tv_denoise_weight : float = 10000,
                    capture_features: bool = False):
        tile_d = int(tile_d)
        tile_h = int(tile_h)
        tile_w = int(tile_w)
        tv_denoise_weight = float(tv_denoise_weight)
        
        #define and load the weights from pretrained seg_model 
        state["segmodel"] = build_and_load_weights_dpt(dims=state["dims"],n_classes=8)

        _eval_widget(tile_h,tile_w,tile_d,tv_denoise_weight,capture_features=capture_features)


        
        

    # --- Double-click similarity callbacks ---
    @magicgui(call_button="Enable Double-Click Similarity")
    def enable_similarity():
        """Attach double-click callback: plots 2D cosine-sim map or 3D NCC map.
        
        Also computes and displays a PCA-RGB feature visualization layer ("feat_rgb")
        if `state["feat"]` is available.

        Expects:
            state["feat"]:
                - 2D mode: ndarray [H, W, C]
                - 3D mode: ndarray [D, H, W, C]
            state["dims"]: 2 or 3
        """


        def on_mouse_double_click(_viewer,event):
            if event.type != "mouse_double_click":
                return
            if state.get("feat", None) is None:
                _ask(viewer, "Similarity", "Run Evaluate with 'capture_features=True' first.")
                return

            y0,x0 = state['offset']

            pos = viewer.cursor.position  # (x, y[, z]) in world coords
            
            print(f"double clicked at {pos}")
            if not pos:
                return

            if state['dims'] ==2:
                pos = pos -[x0,y0]

            if state["dims"] == 2:
                x, y = int(round(pos[0])), int(round(pos[1]))
                feat = state["feat"]  # [H, W, C]
                H, W = feat.shape[:2]
                if not (0 <= x < W and 0 <= y < H):
                    return
                anchor = feat[x, y, :].astype(np.float32)  # [C]
                sim_map = cosine_similarity_map(anchor, feat)  # [H, W]
                if "sim2d" in viewer.layers:
                    del viewer.layers["sim2d"]
                viewer.add_image(sim_map, name="sim2d", colormap="magma", blending="additive",translate=[y0,x0])

            else:
                x, y, z = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))
                feat = state["feat"]  # [D, H, W, C]
                D, H, W = feat.shape[:3]
                if not (0 <= x < W and 0 <= y < H and 0 <= z < D):
                    return
                anchor = feat[z, y, x, :].astype(np.float32)  # [C]
                sim3d = ncc3d(anchor, feat)  # [D, H, W]
                if "ncc3d" in viewer.layers:
                    del viewer.layers["ncc3d"]
                viewer.add_image(sim3d, name="ncc3d", colormap="viridis", blending="additive")

        # hotkey placeholder
        viewer.bind_key("D", lambda v: None)
        # similarity callback
        viewer.mouse_double_click_callbacks.append(on_mouse_double_click)


        _ask(viewer, "Similarity", "Double-click similarity enabled. PCA-RGB layer added if features were present.")

    # Dock the widgets
    viewer.window.add_dock_widget(build_model_widget, area="right")
    viewer.window.add_dock_widget(train_and_eval_widget, area="right")
    viewer.window.add_dock_widget(eval_widget_predefined, area="right")
    viewer.window.add_dock_widget(enable_similarity, area="right")


def main() -> None:
    os.environ.setdefault("NAPARI_ASYNC", "1")
    viewer = napari.Viewer(ndisplay=2)

    # Key binding: toggle predicted segout-like layers
    @viewer.bind_key('v')
    def _toggle_pred(_):
        names = [ln for ln in viewer.layers if any(k in ln.name for k in ("prediction", "segout", "mask", "region", "polygon"))]
        for ln in names:
            viewer.layers[ln].visible = not viewer.layers[ln].visible

    add_ui(viewer)
    napari.run()


if __name__ == "__main__":
    main()
