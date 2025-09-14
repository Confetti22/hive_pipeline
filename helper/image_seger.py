import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
import time
import zarr
import numpy as np
import napari
from magicgui import widgets
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.arch.seg import SegmentationHead,ConvSegHead
from __future__ import annotations

import time
from typing import Callable, Iterable, List, Sequence, Tuple, Union, Optional

import numpy as np
import zarr
import napari
from magicgui import widgets

# ------------------------------- Constants -------------------------------

METHOD_COMPUTE_SIM = "computing_sim"
METHOD_MLP_HEAD    = "mlp_head"
METHOD_CONV_HEAD   = "conv_head"

DEFAULT_STRIDE     = 16
DEFAULT_ROI_SIZE   = (64, 64, 64)
DEFAULT_Z_INFLATE  = 18  # replicate ±18 z-slices
# Alignment base; pick something safely away from dataset min to avoid edge effects
DEFAULT_LB         = (3392 + int(1.5 * DEFAULT_STRIDE),
                      2512 + int(1.5 * DEFAULT_STRIDE),
                      3504 + int(1.5 * DEFAULT_STRIDE))



# ------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------

def _as_callable_or_value(obj):
    """Return obj() if callable, else obj (lets us accept properties or methods)."""
    return obj() if callable(obj) else obj

def upsample_labels_by_stride(
    coarse: np.ndarray, stride: int, pad_left: Sequence[int], pad_right: Sequence[int]
) -> np.ndarray:
    """
    Expand a coarse label grid back to sample space by repeating along each axis,
    then pad to exactly match the sample ROI.

    Args:
        coarse: (Dz, Dy, Dx) int labels in feature space
        stride: lattice stride (e.g., 16)
        pad_left: [lz_pad, ly_pad, lx_pad]
        pad_right: [hz_pad, hy_pad, hx_pad]
    """
    # Repeat via np.repeat (clearer than np.kron for 3D)
    z = np.repeat(coarse, stride, axis=0)
    z = np.repeat(z,      stride, axis=1)
    z = np.repeat(z,      stride, axis=2)

    pad = ((pad_left[0], pad_right[0]),
           (pad_left[1], pad_right[1]),
           (pad_left[2], pad_right[2]))
    return np.pad(z, pad_width=pad, mode="constant", constant_values=0).astype(int)

def compute_stride_alignment(
    roi_offset: Sequence[int], lb: Sequence[int], stride: int
) -> Tuple[List[int], List[int]]:
    """
    Compute the (a) starting voxel indices inside the ROI that land on the feature lattice,
    and (b) the index offsets in feature space.

    Returns
    -------
    vol_start_idx : List[int]
        Start indices inside the sample ROI such that (offset + start - lb) % stride == 0.
    feats_offset  : List[int]
        The corresponding starting indices in feature space.
    """
    vol_start_idx = [stride - (off - base) % stride for off, base in zip(roi_offset, lb)]
    feats_offset  = [int((start + off - base) // stride)
                     for start, off, base in zip(vol_start_idx, roi_offset, lb)]
    return vol_start_idx, feats_offset

def compute_edge_padding(
    sample_shape: Sequence[int], vol_start_idx: Sequence[int], stride: int
) -> Tuple[List[int], List[int]]:
    """
    For a given ROI sample shape and the in-ROI start index, compute left/right padding
    needed after upsampling so the output exactly matches the ROI extent.
    """
    left_pad = [vol_start_idx[0], vol_start_idx[1], vol_start_idx[2]]
    right_pad = []
    for i in range(3):
        rem = (sample_shape[i] - vol_start_idx[i]) % stride
        right_pad.append(rem if rem else stride)
    return left_pad, right_pad

def replicate_nonzero_slices(arr: np.ndarray, n: int) -> np.ndarray:
    """Replicate each non-zero z-slice to ±n neighbors (in-place copy into a fresh array)."""
    D, H, W = arr.shape
    out = arr.copy()
    nonzero_idx = [i for i in range(D) if np.any(arr[i])]
    for idx in nonzero_idx:
        s = max(0, idx - n)
        e = min(D, idx + n + 1)
        out[s:e] = arr[idx]
    return out



# ------------------------------------------------------------------------
# Feature map access (parametrized path; was hard-coded before)
# ------------------------------------------------------------------------

def get_target_feats_map(
    target_shape: Sequence[int],
    roi_offset: Sequence[int],
    lb: Sequence[int],
    stride: int,
    feats_zarr_path: str,
) -> np.ndarray:
    """
    Fetch a (C, Z, Y, X) slice from the global feature zarr and pack as (Z, Y, X, C),
    aligned to the feature lattice and clipped to bounds. Missing areas are zero-filled.
    """
    vol_start_idx, feats_offset = compute_stride_alignment(roi_offset, lb, stride)

    feats_map = zarr.open_array(feats_zarr_path, mode="r")
    C, D, H, W = feats_map.shape

    lz, ly, lx = feats_offset
    hz, hy, hx = [l + s for l, s in zip((lz, ly, lx), target_shape)]

    # Clip to valid bounds
    clipped_lz, clipped_ly, clipped_lx = max(0, lz), max(0, ly), max(0, lx)
    clipped_hz, clipped_hy, clipped_hx = min(D, hz), min(H, hy), min(W, hx)

    index = (slice(None),
             slice(clipped_lz, clipped_hz),
             slice(clipped_ly, clipped_hy),
             slice(clipped_lx, clipped_hx))

    existing = feats_map[index]
    target = np.zeros((C, *target_shape), dtype=feats_map.dtype)

    z0, y0, x0 = clipped_lz - lz, clipped_ly - ly, clipped_lx - lx
    z1, y1, x1 = z0 + (clipped_hz - clipped_lz), y0 + (clipped_hy - clipped_ly), x0 + (clipped_hx - clipped_lx)
    target[:, z0:z1, y0:y1, x0:x1] = existing

    return np.moveaxis(target, 0, -1)  # (Z, Y, X, C)

def map_to_sample_space(
    mapped_seg_out: np.ndarray,
    sample_shape: Sequence[int],
    vol_start_idx: Sequence[int],
    stride: int,
) -> np.ndarray:
    """
    Upsample a coarse (feature-lattice) label to sample space and pad to match ROI.
    """
    mapped_seg_out = np.asarray(mapped_seg_out).squeeze()
    left_pad, right_pad = compute_edge_padding(sample_shape, vol_start_idx, stride)
    return upsample_labels_by_stride(mapped_seg_out, stride, left_pad, right_pad)



def _compute_seg2(label_mask: np.ndarray, feature_map: np.ndarray,  spatial_decay=True,d_sigma=16) -> np.ndarray:
    """
    using similarity to compute seg , compute distance when needed, will be faster

    Parameters
    ----------
    label_mask : np.ndarray
        A 3D array of shape (D, H, W) containing integer class labels for each voxel.
    feature_map : np.ndarray
        A 4D array with dimensions ordered as (D, H, W, C), where C is the number of feature channels.
    spatial_decay : bool, optional
        Whether to apply spatial decay weighting (default: True).
    Returns
    -------
    np.ndarray
        A 3D array of shape (D, H, W) with predicted class labels for each voxel.
    """
    
    print(f"label_mask.shape {label_mask.shape}")

    unique_labels = np.unique(label_mask)
    unique_labels = unique_labels[unique_labels != 0]  # ignore background (if 0)

    if len(unique_labels) < 2:
        return np.zeros(label_mask.shape, dtype=np.uint8)

    D,H, W, C = feature_map.shape
    flat_feats = feature_map.reshape(-1, C)
    num_pixels = flat_feats.shape[0]
    class_similarities = np.full((num_pixels, len(unique_labels)), -np.inf)

    z_coords, y_coords, x_coords = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing='ij'
    )
    all_coords = np.stack([z_coords, y_coords, x_coords], axis=-1).reshape(-1, 3)

    for class_idx, class_label in enumerate(unique_labels):
        class_mask = label_mask == class_label
        if not np.any(class_mask):
            continue

        class_feats = feature_map[class_mask]
        class_indices = np.where(class_mask.reshape(-1))[0]
        class_coords = all_coords[class_indices]

        # Compute cosine similarity
        sim = flat_feats @ class_feats.T

        if spatial_decay:
            voxel_coords = all_coords[:, np.newaxis, :]  # shape: (num_pixels, 1, 3)
            class_coords_exp = class_coords[np.newaxis, :, :]  # shape: (1, num_class_points, 3)
            dists = np.linalg.norm(voxel_coords - class_coords_exp, axis=-1)  # shape: (num_pixels, num_class_points)
            decay_weights = np.exp(-dists**2 / (2*d_sigma**2))
            sim *= decay_weights  # element-wise weighting of similarity

        max_sim = sim.max(axis=1)
        class_similarities[:, class_idx] = max_sim

    # Choose class with the highest similarity
    predicted_classes = np.argmax(class_similarities, axis=1)
    mapped_seg_label = np.array([unique_labels[i] for i in predicted_classes])
    mapped_seg_label = mapped_seg_label.reshape(D,H,W)
    return mapped_seg_label

def replicate_nonzero_slices(arr, n):
    """
    Replicates each non-zero z-slice of arr to n slices before and after.
    
    Parameters:
    - arr: np.ndarray of shape (D, H, W), dtype=int
    - n: int, number of slices to replicate before and after
    
    Returns:
    - arr_copy: np.ndarray with the replicated slices
    """
    D, H, W = arr.shape
    arr_copy = arr.copy()
    
    # Find indices of non-zero slices along the z-axis
    nonzero_z_indices = [i for i in range(D) if np.any(arr[i])]
    
    for idx in nonzero_z_indices:
        start = max(0, idx - n)
        end = min(D, idx + n + 1)
        for i in range(start, end):
            arr_copy[i] = arr[idx]
    
    return arr_copy

def _seg_via_mlp_head(user_mask, feature_map, num_epochs=100, lr=1e-3, return_prob=False):
    """
    Args:
        user_mask: numpy of shape (D, H, W) or (H, W), where labeled voxels have integer class labels >= 0.
        feature_map: numpy of shape (D, H, W, C) or (H, W, C)
    
    Returns:
        predicted_mask: numpy of shape (D, H, W) or (H, W), with predicted class labels (int64)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    feature_map = torch.from_numpy(feature_map)
    user_mask = torch.from_numpy(user_mask)

    if feature_map.dim() == 4:
        D, H, W, C = feature_map.shape
        coords = torch.nonzero(user_mask > 0, as_tuple=False)  # [N, 3]
        z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
        prompt_features = feature_map[z, y, x]  # [N, C]
    elif feature_map.dim() == 3:
        H, W, C = feature_map.shape
        coords = torch.nonzero(user_mask > 0, as_tuple=False)  # [N, 2]
        y, x = coords[:, 0], coords[:, 1]
        prompt_features = feature_map[y, x]  # [N, C]
    else:
        raise ValueError("feature_map must be 3D or 4D")

    labels = user_mask[tuple(coords.T)] - 1  # Convert to 0-based index
    num_classes = labels.max().item() + 1

    if num_classes < 2:
        raise ValueError("Need at least 2 labeled classes in the mask.")

    head = SegmentationHead(C, num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    prompt_features = prompt_features.to(device)
    prompt_labels = labels.to(device)

    for epoch in tqdm(range(num_epochs)):
        head.train()
        optimizer.zero_grad()
        logits = head(prompt_features)
        loss = loss_fn(logits, prompt_labels)
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item():.4f}")

    # Predict over full volume/image
    flat_features = feature_map.reshape(-1, C).to(device)
    with torch.no_grad():
        head.eval()
        logits = head(flat_features)
        probs = F.softmax(logits, dim=1)

    if return_prob:
        prob_vol = probs.reshape((-1, num_classes)).reshape(*feature_map.shape[:-1], num_classes)
        if feature_map.dim() == 4:
            prob_vol = prob_vol.permute(3, 0, 1, 2)  # [K, D, H, W]
        else:
            prob_vol = prob_vol.permute(2, 0, 1)      # [K, H, W]
        return prob_vol.detach().cpu().numpy()
    else:
        pred_mask = torch.argmax(probs, dim=1).reshape(feature_map.shape[:-1]) + 1
        return pred_mask.detach().cpu().numpy()



def _seg_via_conv_head(user_input_label, feature_map, num_epochs=100, lr=1e-3, return_prob=False):
    """
    Args:
        user_input_label: numpy of shape (D, H, W) or (H, W), labels >= 1
        feature_map: numpy of shape (D, H, W, C) or (H, W, C)
    Returns:
        predicted_mask or probabilities of shape (D, H, W) or (H, W)
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    feature_map = torch.from_numpy(feature_map).float()
    user_input_label = torch.from_numpy(user_input_label).long()

    is_3d = feature_map.dim() == 3  # (H, W, C)
    is_4d = feature_map.dim() == 4  # (D, H, W, C)

    if is_4d:
        D, H, W, C = feature_map.shape
        feat = feature_map.permute(3, 0, 1, 2).to(device)  # [C, D, H, W]
        labels = user_input_label.to(device) - 1  # 0-based
        mask = (labels >= 0)
        spatial_shape = (D, H, W)
    elif is_3d:
        H, W, C = feature_map.shape
        feat = feature_map.permute(2, 0, 1).unsqueeze(1).to(device)  # [C, 1, H, W]
        labels = user_input_label.to(device) - 1  # 0-based
        mask = (labels >= 0)
        spatial_shape = (H, W)
    else:
        raise ValueError("feature_map must be 3D or 4D")

    num_classes = labels.max().item() + 1
    if num_classes < 2:
        raise ValueError("Need at least 2 labeled classes in the mask.")

    head = ConvSegHead(C, num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(num_epochs)):
        head.train()
        optimizer.zero_grad()
        logits = head(feat)  # [K, D, H, W] or [K, 1, H, W]

        logits = logits if is_4d else logits.squeeze(1)  # remove dummy D=1 for 2D

        # Permute logits to [D, H, W, K] or [H, W, K]
        logits_flat = logits.permute(*range(1, logits.ndim), 0)[mask]  # [N, K]
        labels_flat = labels[mask]

        loss = loss_fn(logits_flat, labels_flat)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f"Conv head epoch {epoch}, loss: {loss.item():.4f}")

    # Inference
    head.eval()
    with torch.no_grad():
        logits = head(feat)
        logits = logits if is_4d else logits.squeeze(1)
        probs = F.softmax(logits, dim=0)  # [K, D, H, W] or [K, H, W]

    if return_prob:
        return probs.detach().cpu().numpy()
    else:
        pred = torch.argmax(probs, dim=0) + 1  # back to 1-based label
        return pred.detach().cpu().numpy()


# --- UI Controller Class ---

class SimpleSeger2(widgets.Container):
    """
    Napari UI controller that lets a user paint sparse 3D labels in a ROI and
    run prompt-based segmentation using precomputed features.

    Overview
    --------
    The tool synchronizes with an external ROI controller (`simple_viewer`) to
    know the current cube-of-interest (offset and size). The user paints
    integer labels (>=1) into a `Label` layer, then triggers segmentation
    via one of three backends:

      1) "computing_sim"  -> Direct feature similarity (optionally with spatial decay)
      2) "mlp_head"       -> A tiny MLP trained on the painted points' features
      3) "conv_head"      -> A light Conv head trained over the ROI feature map

    Pipeline (per segmentation)
    ---------------------------
    1) Read the ROI offset/size from `simple_viewer`.
    2) Inflate (replicate) non-empty z-slices in the user label to thicken sparse
       scribbles along Z. This stabilizes training and similarity votes.
    3) Downsample/mosaic the user label into feature space by aligning to the
       global feature stride and cropping to the ROI's feature grid.
    4) Fetch the matching slice of the global feature map (D, H, W, C) for the ROI.
    5) Run the chosen backend to produce a coarse label in feature space.
    6) Upsample the coarse label back to sample space (via stride-aware zoom +
       padding to exactly fill the sample ROI bounds).
    7) Write the final segmentation into the `Segout` layer.

    Key Concepts
    ------------
    - Feature stride alignment: The global feature volume is typically stride-16
      w.r.t. the original image grid. Offsets and shapes must be snapped to this
      lattice to address the correct feature voxels.
    - Label inflation along Z: user scribbles are often 2D planes; replicating
      nearby slices (±n) gives 3D context without asking the user to paint
      every slice.
    - Stateless backends: each `Seg` click performs a fresh mapping/training
      pass on the current ROI and labels only; nothing is cached globally.

    Layers
    ------
    - `Label`  : user-painted integer labels (>= 1), brush mode enabled.
    - `Segout` : model's predicted segmentation for the current ROI.

    Buttons
    -------
    - "Seg"     : run segmentation with the selected method.
    - "Clear"   : clear both `Label` and `Segout` in the current ROI.
    - "Undo"    : revert to the previous `Label` and `Segout` buffers.

    Parameters
    ----------
    viewer1 : napari.Viewer
        The main Napari viewer hosting the label and segmentation layers.
    viewer2 : napari.Viewer
        (Currently unused but kept for compatibility.)
    simple_viewer : object
        A controller exposing `get_roi_offset()` and `get_roi_size()` to define
        the active ROI. These may be callables or properties.

    Defaults / Tunables
    -------------------
    - stride : int = 16  (feature stride)
    - lb     : List[int]  (global alignment base per axis; set ~1.5 * stride
                           away from dataset lower bounds to avoid edge effects)
    - roi_size : List[int] = [64, 64, 64]
    - z_inflate : int = 18 (± slices to replicate around non-empty z-planes)

    Notes
    -----
    - The global feature map is retrieved on demand for the aligned ROI via
      `get_target_feats_map`. Its path should be configurable (not hard-coded).
    - The MLP/Conv heads are intentionally tiny and train per-ROI on the fly;
      they do not modify the upstream feature extractor.
    """
    def __init__(
        self,
        viewer1: napari.Viewer,
        viewer2: napari.Viewer,  # kept for compatibility
        simple_viewer,
        *,
        feats_zarr_path: str = "/home/confetti/data/t1779/mlp_feats.zarr",
        stride: int = DEFAULT_STRIDE,
        lb: Sequence[int] = DEFAULT_LB,
        roi_size: Sequence[int] = DEFAULT_ROI_SIZE,
        z_inflate: int = DEFAULT_Z_INFLATE,
    ) -> None:
        super().__init__()
        self.viewer1 = viewer1
        self.simple_viewer = simple_viewer
        self.feats_zarr_path = feats_zarr_path

        self.stride = int(stride)
        self.lb = list(map(int, lb))
        self.roi_size = tuple(map(int, roi_size))
        self.z_inflate = int(z_inflate)

        self._init_label_buffers()
        self._setup_layers()
        self._setup_controls()
        self._register_callbacks()

    # -------------------------- layer & UI setup --------------------------

    def _init_label_buffers(self) -> None:
        shape = tuple(self.roi_size)
        self.last_seg_data   = np.zeros(shape, dtype=np.uint8)
        self.last_label_data = np.zeros(shape, dtype=np.uint8)
        self.current_label_data = np.zeros(shape, dtype=np.uint8)

    def _setup_layers(self) -> None:
        zero = np.zeros(self.roi_size, dtype=np.uint8)
        self.label_layer  = self.viewer1.add_labels(zero, name="Label")
        self.segout_layer = self.viewer1.add_labels(zero, name="Segout")
        self.label_layer.brush_size = 30
        self.label_layer.mode = "PAINT"
        self.viewer1.layers.selection = [self.label_layer]

    def _setup_controls(self) -> None:
        self.method_selector = widgets.ComboBox(
            choices=[METHOD_COMPUTE_SIM, METHOD_MLP_HEAD, METHOD_CONV_HEAD],
            value=METHOD_COMPUTE_SIM,
            label="Segmentation Method",
        )
        self.btn_seg   = widgets.PushButton(text="Seg")
        self.btn_clear = widgets.PushButton(text="Clear")
        self.btn_undo  = widgets.PushButton(text="Undo")

        self.btn_seg.clicked.connect(self.run_seg)
        self.btn_clear.clicked.connect(self.clear_labels)
        self.btn_undo.clicked.connect(self.undo_labels)

        self.extend([self.method_selector, self.btn_seg, self.btn_clear, self.btn_undo])

    def _register_callbacks(self) -> None:
        # Refresh ROI-sized arrays whenever the ROI layer changes
        self.simple_viewer.roi_layer.events.data.connect(self._prepare_for_new_roi)

    # ------------------------------ actions ------------------------------

    def _prepare_for_new_roi(self) -> None:
        roi_size = self.read_roi_size()
        self.label_layer.data  = np.zeros(roi_size, dtype=np.uint8)
        self.segout_layer.data = np.zeros(roi_size, dtype=np.uint8)
        self.current_label_data = np.zeros(roi_size, dtype=np.uint8)

    def run_seg(self) -> None:
        self._backup_current_state()

        roi_offset = self.read_roi_offset()
        roi_size   = self.read_roi_size()
        label_data = self.label_layer.data.copy()

        method = self.method_selector.value

        # Step 1: inflate sparse labels along Z
        inflated = replicate_nonzero_slices(label_data, n=self.z_inflate)

        # Step 2: compute alignment and downsample label into feature grid
        vol_start_idx, _ = compute_stride_alignment(roi_offset, self.lb, self.stride)
        mapped_label = inflated[
            vol_start_idx[0]::self.stride,
            vol_start_idx[1]::self.stride,
            vol_start_idx[2]::self.stride,
        ][:-1, :-1, :-1]  # keep consistent with feature tiling

        # Step 3: fetch aligned feature subvolume
        feats = get_target_feats_map(
            mapped_label.shape,
            roi_offset=roi_offset,
            lb=self.lb,
            stride=self.stride,
            feats_zarr_path=self.feats_zarr_path,
        )

        # Step 4: run backend
        start = time.time()
        if method == METHOD_COMPUTE_SIM:
            mapped_seg = _compute_seg2(
                label_mask=mapped_label, feature_map=feats, spatial_decay=False
            )
        elif method == METHOD_MLP_HEAD:
            mapped_seg = _seg_via_mlp_head(
                user_mask=mapped_label, feature_map=feats, num_epochs=2000
            )
        elif method == METHOD_CONV_HEAD:
            mapped_seg = _seg_via_conv_head(
                user_input_label=mapped_label, feature_map=feats, num_epochs=2000
            )
        else:
            print(f"[WARN] Unknown method: {method}")
            return
        print(f"[INFO] Segmentation time: {time.time() - start:.2f}s")

        # Step 5: upsample back to sample space
        seg_out = map_to_sample_space(mapped_seg, roi_size, vol_start_idx, self.stride)

        # Step 6: display
        self.segout_layer.data = seg_out
        self.viewer1.layers.selection = [self.label_layer]

    def clear_labels(self) -> None:
        shape = self.label_layer.data.shape
        self.label_layer.data  = np.zeros(shape, dtype=np.uint8)
        self.segout_layer.data = np.zeros(shape, dtype=np.uint8)
        self.viewer1.layers.selection = [self.label_layer]

    def undo_labels(self) -> None:
        self.label_layer.data  = self.last_label_data
        self.segout_layer.data = self.last_seg_data
        self.viewer1.layers.selection = [self.label_layer]

    # ------------------------------ helpers ------------------------------

    def _backup_current_state(self) -> None:
        self.last_label_data     = self.current_label_data.copy()
        self.last_seg_data       = self.segout_layer.data.copy()
        self.current_label_data  = self.label_layer.data.copy()

    def read_roi_offset(self) -> Tuple[int, int, int]:
        # accept either property or method on simple_viewer
        return tuple(_as_callable_or_value(self.simple_viewer.get_roi_offset))

    def read_roi_size(self) -> Tuple[int, int, int]:
        return tuple(_as_callable_or_value(self.simple_viewer.get_roi_size))
    

        return self.simple_viewer.get_roi_size