#!/usr/bin/env python3
"""
Test script to extract Inception features and train a tiny MLP seg head.

Pipeline
--------
1) Load the t1779 ROI + user labels via `load_t1779`.
2) Extract Inception-V3 avgpool features using the same stride/window settings
   as the interactive viewer (TraverseDataset2d, stride=8, win_size=128).
3) Visualize the feature map with a 3-channel PCA projection.
4) Downscale labels to feature resolution, train the MLP head (see
   `_seg_via_mlp_head` in helper/image_seger.py), and upscale predictions.
5) Save feature npy, PCA PNG, predicted mask (tif + npy), and a small report.

Run inside the `pytorch` conda env:
    python eval_inception_feature_for_segmentation.py --output-dir results/eval_inception_seg
"""

import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile as tif
import torch
from scipy.ndimage import maximum_filter, zoom
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import Inception_V3_Weights

from confettii.feat_extract import TraverseDataset2d, get_feature_list
from confettii.plot_helper import three_pca_as_rgb_image
from helper.image_seger import _seg_via_mlp_head


def _get_path_map():
    return {
        "visa": {
            "roi": "visa_1536_1536_12.tif",
            "label": "visa_label.tif",
            "mask": "visa_mask.tif",
        },
        "hp": {
            "roi": "hp_off7000_2962_4452_sieze1536_1536_12.tif",
            "label": "hp_label.tif",
            "mask": None,
        },
        "7N": {
            "roi": "vii_1536_1536_83.tif",
            "label": "7N_label.tif",
            "mask": "7N_mask.tif",
        },
        # Widefield set
        "2_1": {
            "roi": "wf_hp_1536_1536.tif",
            "label": "wf_hp_label.tif",
            "mask": None,
        },
        "2_2": {
            "roi": "wf_viin_1536_1536.tif",
            "label": "wf_7n_label.tif",
            "mask": None,
        },
        "2_3": {
            "roi": "wf_visa_1536_1536.tif",
            "label": "wf_visa_label.tif",
            "mask": "wf_visa_mask.tif",
        },
        # DK set
        "3_1": {
            "roi": "dk_hp_roi.tif",
            "label": "dk_hp_label.tif",
            "mask": None,
        },
        "3_2": {
            "roi": "dk_vii_roi.tif",
            "label": "dk_7N_label.tif",
            "mask": None,
        },
        "3_3": {
            "roi": "dk_vis_roi.tif",
            "label": "dk_vis_label.tif",
            "mask": "dk_vis_mask.tif",
        },
    }


def load_t1779(region_key: str = "7N"):
    """
    Minimal copy of load_t1779 from interactive_svc_single_viewer to avoid napari imports.
    Returns a 2D max-projection ROI, its label map, and optional mask.
    """
    path_map = _get_path_map()
    if region_key not in path_map:
        raise ValueError(f"Unknown region_key '{region_key}'. Available: {list(path_map.keys())}")

    parent_dir = Path("/home/confetti/data/t1779/scenes")
    results_dir = parent_dir / "results"
    roi_path = parent_dir / path_map[region_key]["roi"]
    label_name = path_map[region_key]["label"]
    label_path = None
    if label_name is not None:
        candidate = parent_dir / label_name
        label_path = candidate if candidate.exists() else results_dir / label_name
    mask_path = (
        parent_dir / path_map[region_key]["mask"] if path_map[region_key]["mask"] is not None else None
    )

    roi_vol = tif.imread(roi_path)
    if len(roi_vol.shape) == 3 and roi_vol.shape[-1] != 3:
        roi = np.max(roi_vol, axis=0)
    else:
        roi = roi_vol
    roi = np.squeeze(roi)
    if roi.ndim not in (2, 3):
        raise RuntimeError(f"Unexpected ROI shape {roi.shape} for {region_key}")

    if label_path is not None and not label_path.exists():
        raise FileNotFoundError(f"Label not found at {label_path}")
    label = tif.imread(label_path) if label_path is not None else None
    label = np.squeeze(label) if label is not None else None

    mask = tif.imread(mask_path) if mask_path is not None else None
    mask = np.squeeze(mask) if mask is not None else None

    return roi, label, mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Inception features + MLP seg head on t1779.")
    parser.add_argument("--region-key", default="7N", help="Region key for load_t1779 (see interactive_svc_single_viewer.get_path_map).")
    parser.add_argument("--stride", type=int, default=8, help="Sliding window stride for TraverseDataset2d.")
    parser.add_argument("--win-size", type=int, default=128, help="Sliding window size for TraverseDataset2d.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for feature extraction.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for the MLP head.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the MLP head.")
    parser.add_argument("--output-dir", default="results/eval_inception_feature_for_segmentation", help="Directory to store outputs.")
    parser.add_argument("--device", default='cuda', help="Override device (cpu/cuda). Defaults to cuda if available.")
    return parser.parse_args()


def _ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def _resolve_device(user_device: str | None) -> str:
    """Pick a usable device, testing CUDA before returning it."""
    if user_device:
        if user_device.startswith("cuda"):
            try:
                torch.zeros(1).to(user_device)
                return user_device
            except Exception as exc:
                print(f"Requested CUDA device '{user_device}' unavailable ({exc}); falling back to cpu.")
                return "cpu"
        return user_device

    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except Exception as exc:
            print(f"CUDA reported available but failed to initialize ({exc}); using cpu.")
            return "cpu"
    return "cpu"


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img_min = float(img.min())
    img_max = float(img.max())
    denom = img_max - img_min
    if denom == 0:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - img_min) / (denom + 1e-8)
    return (scaled * 255).astype(np.uint8)


def extract_inception_features(
    roi: np.ndarray,
    device: str,
    stride: int,
    win_size: int,
    batch_size: int,
    extract_layer_name: str = "avgpool",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mirror the interactive viewer's Inception feature extraction.

    Returns
    -------
    feats_map : np.ndarray
        Feature map shaped (H_feat, W_feat, C).
    rgb_img : np.ndarray
        Normalized RGB image fed to the feature extractor.
    """
    print("Begin prepare for Inception")
    try:
        inception_model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    except Exception as exc:
        print(f"Falling back to random InceptionV3 weights due to: {exc}")
        inception_model = models.inception_v3(weights=None, aux_logits=True)
    inception_model.eval().to(device)

    normalized_img = _normalize_to_uint8(roi)
    if normalized_img.ndim == 2:
        rgb_img = np.stack([normalized_img] * 3, axis=-1)
    elif normalized_img.ndim == 3 and normalized_img.shape[-1] == 3:
        rgb_img = normalized_img
    else:
        raise ValueError(f"Unexpected ROI shape for Inception input: {normalized_img.shape}")

    dataset = TraverseDataset2d(rgb_img, stride=stride, win_size=win_size)
    out_shape = dataset._get_sample_shape()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    start = time.time()
    with torch.no_grad():
        feats_list = get_feature_list(device, inception_model, loader, extract_layer_name=extract_layer_name)
    print(f"Extracting feats in Inception took {time.time() - start:.3f}s")

    feats_map = feats_list.reshape((*out_shape, -1))
    print(f"End prepare for Inception | feats_map shape={feats_map.shape}")

    del inception_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return feats_map, rgb_img


def visualize_feats(
    feats_map: np.ndarray,
    save_path: str,
    target_shape: Tuple[int, int] | None = None,
    mask: np.ndarray | None = None,
) -> None:
    def _resize_mask(src: np.ndarray, dst_shape: Tuple[int, int]) -> np.ndarray:
        if src.shape == dst_shape:
            return src.astype(bool, copy=False)
        zoom_factors = (dst_shape[0] / src.shape[0], dst_shape[1] / src.shape[1])
        return zoom(src.astype(float), zoom=zoom_factors, order=0) > 0.5

    h_f, w_f, c_f = feats_map.shape
    start = time.time()
    mask_feats = _resize_mask(mask, (h_f, w_f)) if mask is not None else None
    if mask_feats is None:
        rgb_feats_map = three_pca_as_rgb_image(feats_map.reshape(-1, c_f), final_image_shape=(h_f, w_f))
    elif mask_feats.any():
        masked_flat = feats_map[mask_feats]
        masked_rgb = three_pca_as_rgb_image(masked_flat, final_image_shape=(masked_flat.shape[0],))
        rgb_feats_map = np.zeros((h_f * w_f, 3), dtype=masked_rgb.dtype)
        rgb_feats_map[mask_feats.reshape(-1)] = masked_rgb.reshape(-1, 3)
        rgb_feats_map = rgb_feats_map.reshape(h_f, w_f, 3)
    else:
        rgb_feats_map = np.zeros((h_f, w_f, 3), dtype=np.float32)

    if target_shape is not None and (h_f != target_shape[0] or w_f != target_shape[1]):
        zoom_factors = (target_shape[0] / h_f, target_shape[1] / w_f, 1.0)
        upscaled = zoom(rgb_feats_map, zoom=zoom_factors, order=1)
        if upscaled.shape[:2] != target_shape:
            trimmed = upscaled[: target_shape[0], : target_shape[1], :]
            adjusted = np.zeros((*target_shape, upscaled.shape[2]), dtype=upscaled.dtype)
            adjusted[: trimmed.shape[0], : trimmed.shape[1], :] = trimmed
            rgb_feats_map = adjusted
        else:
            rgb_feats_map = upscaled
    if mask is not None:
        final_mask = _resize_mask(mask, rgb_feats_map.shape[:2])
        rgb_feats_map *= final_mask[..., None]
    print(f"PCA projection finished in {time.time() - start:.3f}s; saving to {save_path}")
    tif.imwrite(save_path, (rgb_feats_map * 255).astype(np.uint8))


def downscale_labels_to_feats(label: np.ndarray, feats_shape: Tuple[int, int]) -> np.ndarray:
    zoom_factors = [x / y for x, y in zip(feats_shape, label.shape)]
    return zoom(label, zoom=zoom_factors, order=0)


def upscale_pred_to_img(pred: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    zoom_factors = [y / x for x, y in zip(pred.shape, target_shape)]
    up = zoom(pred, zoom=zoom_factors, order=0)
    if up.shape != target_shape:
        # Adjust for minor rounding differences from zoom
        trimmed = up[: target_shape[0], : target_shape[1]]
        out = np.zeros(target_shape, dtype=up.dtype)
        out[: trimmed.shape[0], : trimmed.shape[1]] = trimmed
        return out
    return up


def compute_labeled_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    valid = gt > 0
    if not np.any(valid):
        return float("nan")
    return float((pred[valid] == gt[valid]).sum() / valid.sum())


def main() -> None:
    args = parse_args()
    out_dir = _ensure_output_dir(args.output_dir)
    device = _resolve_device(args.device)

    roi, label, mask = load_t1779(args.region_key)
    if label is None:
        raise RuntimeError("No user label found from load_t1779; cannot train seg head.")

    roi = np.asarray(roi)
    label = np.asarray(label, dtype=np.uint8)
    if mask is not None:
        label = np.where(np.asarray(mask, dtype=bool), label, 0)

    feats_map, rgb_img = extract_inception_features(
        roi=roi,
        device=device,
        stride=args.stride,
        win_size=args.win_size,
        batch_size=args.batch_size,
        extract_layer_name="avgpool",
    )

    feats_save = os.path.join(out_dir, "inception_feats.npy")
    np.save(feats_save, feats_map)
    print(f"Saved feature map to {feats_save}")

    visualize_feats(
        feats_map,
        os.path.join(out_dir, "inception_feats_pca.tif"),
        target_shape=roi.shape[:2],
        mask=mask,
    )


    # Downscale labels to feature lattice and train the tiny MLP head.
    scale_y = label.shape[0] / feats_map.shape[0]
    scale_x = label.shape[1] / feats_map.shape[1]
    dilation_radius = max(1, int(np.ceil(max(scale_y, scale_x) * 0.5)))
    dilated_label = maximum_filter(label, size=2 * dilation_radius + 1, mode="nearest")
    label_ds = downscale_labels_to_feats(dilated_label, feats_map.shape[:2]).astype(np.uint8)
    label_ds_save_npy = os.path.join(out_dir, "label_downscaled.npy")
    label_ds_save_tif = os.path.join(out_dir, "label_downscaled.tif")
    np.save(label_ds_save_npy, label_ds)
    tif.imwrite(label_ds_save_tif, label_ds)
    print(f"Saved downscaled label to {label_ds_save_tif} and {label_ds_save_npy}")
    unique = np.unique(label_ds[label_ds > 0])
    if len(unique) < 2:
        raise RuntimeError(f"Need at least 2 labeled classes after downscaling; got {unique}.")

    print(f"Training MLP head on {len(np.nonzero(label_ds)[0])} labeled pixels at feature resolution {label_ds.shape}")
    pred_feats_space = _seg_via_mlp_head(label_ds, feats_map, num_epochs=args.epochs, lr=args.lr, return_prob=False)

    pred_img_space = upscale_pred_to_img(pred_feats_space, roi.shape[:2]).astype(np.uint8)
    acc = compute_labeled_accuracy(pred_img_space, label)
    print(f"Labeled-pixel accuracy (upsampled to image space): {acc:.4f}")

    pred_save_npy = os.path.join(out_dir, "pred_mask.npy")
    pred_save_tif = os.path.join(out_dir, "pred_mask.tif")
    np.save(pred_save_npy, pred_img_space)
    tif.imwrite(pred_save_tif, pred_img_space.astype(np.uint8))
    print(f"Saved predicted mask to {pred_save_tif} and {pred_save_npy}")

    report_path = os.path.join(out_dir, "run_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"region_key: {args.region_key}\n")
        f.write(f"roi_shape: {roi.shape}\n")
        f.write(f"label_shape: {label.shape}\n")
        f.write(f"feats_shape: {feats_map.shape}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"stride: {args.stride}\n")
        f.write(f"win_size: {args.win_size}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"device: {device}\n")
        f.write(f"labeled_pixel_accuracy: {acc}\n")
    print(f"Wrote run report to {report_path}")


if __name__ == "__main__":
    main()
