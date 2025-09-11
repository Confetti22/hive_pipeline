#%%
#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
from typing import Iterable, Tuple
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import zarr
from tifffile import TiffFile, memmap
from confettii.feat_extract import SlidingWindowND
from typing import Tuple, Union

# -----------------------------------------------------------------------------
#                       I / O   a n d   v o l u m e   r e a d e r
# -----------------------------------------------------------------------------

from pathlib import Path
import re
from typing import Tuple
import numpy as np
from tifffile import memmap as tiff_memmap
import tifffile as tif


# ----------------------------- new/updated VolumeReader -----------------------------
def _safe_read_tiff(path: Path) -> np.ndarray:
    """Read a TIFF slice robustly; memmap if possible, else full read."""
    try:
        return tiff_memmap(path)  # fast path (no decompression if possible)
    except Exception:
        # Non-mappable (compressed/tiled) → fall back to full read (decompress)
        return tif.imread(str(path))  # returns np.ndarray

def _to_gray(arr_hw3: np.ndarray) -> np.ndarray:
    # simple luminance approx; input (H,W,3) -> (H,W,1)
    g = (0.2989*arr_hw3[...,0] + 0.5870*arr_hw3[...,1] + 0.1140*arr_hw3[...,2]).astype(arr_hw3.dtype)
    return g[..., None]

class VolumeReader:
    """Normalize all inputs to (D, H, W, C).

    Supports:
      1) single `.ims`  (assumes one selected channel -> C=1)
      2) single `.tif/.tiff` (2-D -> (1,H,W,1) or (1,H,W,3); 3-D -> (D,H,W,1))
      3) directory of 2-D `.tif/.tiff` -> stacked along Z (D,H,W,1 or 3)

    Usage:
        with VolumeReader(path, img_channel=1, ims_channel=0) as vol:
            block = vol.read_block(offset=(z,y,x), size=(d,h,w))   # -> (d,h,w,c)
    """
    def __init__(self, path: Union[str, Path], *, img_channel: int = 1, ims_channel: int = 0):
        assert img_channel in (1, 3), "img_channel must be 1 or 3"
        self.path = Path(path)
        self.desired_c = img_channel
        self.ims_channel = ims_channel

        self._is_dir = self.path.is_dir()
        self._dir_files: list[Path] = []
        self._handle = None    # ims object or ndarray memmap
        self._shape: Tuple[int,int,int,int] | None = None  # (D,H,W,C)

    @staticmethod
    def _natural_key(p: Path):
        parts = re.split(r'(\d+)', p.name)
        return [int(s) if s.isdigit() else s.lower() for s in parts]

    def __enter__(self):
        if self._is_dir:
            self._dir_files = sorted(
                [p for p in self.path.iterdir() if p.suffix.lower() in {'.tif', '.tiff'}],
                key=self._natural_key
            )
            if not self._dir_files:
                raise ValueError(f"No .tif/.tiff files found in directory: {self.path}")

            first = _safe_read_tiff(self._dir_files[0])
            if first.ndim == 2:
                H, W = map(int, first.shape)
                C = self.desired_c
            elif first.ndim == 3:
                H, W, Cin = map(int, first.shape)
                # force to desired (1 or 3)
                C = 3 if self.desired_c == 3 else 1
            else:
                raise ValueError(f"Directory mode expects 2-D or 2-D+channels TIFFs, got ndim={first.ndim}")

            D = len(self._dir_files)
            self._shape = (D, H, W, C)
            return self

        # single file
        suffix = self.path.suffix.lower()
        if suffix == ".ims":
            from helper.image_reader import Ims_Image
            self._handle = Ims_Image(str(self.path), channel=self.ims_channel)
            d,h,w = (int(x) for x in self._handle.rois[0][3:])  # (D,H,W)
            self._shape = (d, h, w, 1)  # one selected channel
        elif suffix in {".tif", ".tiff"}:
            try:
                arr = tiff_memmap(self.path)
            except Exception:
                arr = tif.imread(str(self.path))  # fallback full read

            if arr.ndim == 2:     # (H,W) -> (1,H,W,C)
                H, W = map(int, arr.shape)
                C = self.desired_c
                self._handle = arr  # keep reference
                self._shape = (1, H, W, C)
            elif arr.ndim == 3:
                # Could be (D,H,W) OR (H,W,C). We decide by last-dim size.
                if arr.shape[-1] in (1,3):  # treat as (H,W,C)
                    H, W, Cin = map(int, arr.shape)
                    C = 3 if self.desired_c == 3 else 1
                    self._handle = arr
                    self._shape = (1, H, W, C)
                else:  # (D,H,W)
                    D, H, W = map(int, arr.shape)
                    self._handle = arr
                    self._shape = (D, H, W, 1)
            else:
                raise ValueError(f"Unsupported TIFF ndim={arr.ndim}, expected 2 or 3.")
        else:
            raise ValueError(f"Unsupported volume format: {self.path.suffix}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None and hasattr(self._handle, "close"):
            self._handle.close()

    @property
    def shape(self) -> Tuple[int,int,int,int]:
        if self._shape is None:
            raise RuntimeError("VolumeReader not opened. Use as a context manager.")
        return self._shape  # (D,H,W,C)

    def _ensure_c(self, arr_hw_or_hw3: np.ndarray) -> np.ndarray:
        """Make per-slice array shape (H,W,C) with C in {1,3}."""
        if arr_hw_or_hw3.ndim == 2:
            arr = arr_hw_or_hw3[..., None]  # (H,W,1)
        elif arr_hw_or_hw3.ndim == 3:
            if arr_hw_or_hw3.shape[-1] in (1,3):
                arr = arr_hw_or_hw3
            else:
                raise ValueError("Ambiguous 3-D TIFF with last-dim not 1/3.")
        else:
            raise ValueError("Unexpected slice shape.")

        if self.desired_c == 1:
            if arr.shape[-1] == 1:
                return arr
            return _to_gray(arr)  # (H,W,1)
        else:  # desired 3
            if arr.shape[-1] == 3:
                return arr
            return np.repeat(arr, 3, axis=-1)  # (H,W,3)
    
    def read_block(self, *, offset: Tuple[int,int,int], size: Tuple[int,int,int]) -> np.ndarray:
        """Return (d,h,w,c) normalized block from (D,H,W,C) space."""
        z0, y0, x0 = offset
        dz, dh, dw = size

        if self._is_dir:
            D, H, W, C = self.shape
            z1 = min(z0 + dz, D)
            if z0 >= D:
                return np.zeros((0, min(dh, H - y0), min(dw, W - x0), C), dtype=np.float32)

            slices = []
            for zi in range(z0, z1):
                raw = _safe_read_tiff(self._dir_files[zi])  # handles compressed/tiled
                if raw.ndim == 2:
                    hwc = self._ensure_c(np.asarray(raw))
                elif raw.ndim == 3:
                    hwc = self._ensure_c(np.asarray(raw))
                else:
                    raise ValueError(f"Slice ndim={raw.ndim} not supported.")

                patch = hwc[y0:y0+dh, x0:x0+dw, :]
                slices.append(patch)
            if not slices:
                return np.zeros((0, dh, dw, self.shape[-1]), dtype=np.float32)
            return np.stack(slices, axis=0).astype(np.float32)

        # single files
        if hasattr(self._handle, "from_roi"):  # IMS → returns (d,h,w)
            coords = np.array([z0, y0, x0, dz, dh, dw], dtype=np.int64)
            arr = self._handle.from_roi(coords=coords, level=0)   # (d,h,w)
            arr = arr[..., None]                                  # (d,h,w,1)
            return arr.astype(np.float32)

        arr = self._handle  # tiff memmap
        if arr.ndim == 2:
            hwc = self._ensure_c(np.asarray(arr))
            patch = hwc[y0:y0+dh, x0:x0+dw, :]
            return patch[None, ...].astype(np.float32)  # (1,h,w,c)
        elif arr.ndim == 3:
            # either (D,H,W) or (H,W,C)
            if arr.shape[-1] in (1,3):  # (H,W,C)
                hwc = self._ensure_c(np.asarray(arr))
                patch = hwc[y0:y0+dh, x0:x0+dw, :]
                return patch[None, ...].astype(np.float32)
            else:  # (D,H,W)
                dblk = np.asarray(arr[z0:z0+dz, y0:y0+dh, x0:x0+dw])[..., None]
                return dblk.astype(np.float32)
        else:
            raise ValueError("Unexpected TIFF ndim.")
# -----------------------------------------------------------------------------
#                  C o o r d i n a t e   m a p p i n g   h e l p e r
# -----------------------------------------------------------------------------
# -------------------------- helper: make 3D ROI from ints --------------------------
"""
Quick sanity checklist
	•	2-D single TIFF (H,W):
→ normalized to (1,H,W,C), roi3=(1,r,r), stride3=(1,s,s)
	•	3-D single TIFF (D,H,W):
→ normalized to (D,H,W,1), roi3=(r,r,r), stride3=(s,s,s)
	•	Dir of 2-D grayscale TIFFs:
→ stacked to (D,H,W,1), roi3=(1,r,r), stride3=(1,s,s)
	•	Dir of 2-D RGB TIFFs:
→ stacked to (D,H,W,3) (or gray if img_channel=1), same roi3/stride3 as above
	•	Model input: always (B,C,D,H,W); if using a 2-D backbone, you can squeeze D=1.
"""

def _roi_tuple_from_ints(kind: str, roi_size: int, roi_stride: int) -> tuple[tuple[int,int,int], tuple[int,int,int]]:
    """
    kind ∈ {"2d_single","3d_single","dir_2d"}
      - "2d_single": single 2-D image (TIFF/HWC or HW) -> use (1, r, r) / (1, s, s)
      - "3d_single": single 3-D volume (DHW) -> use (r, r, r) / (s, s, s)
      - "dir_2d":    directory of per-slice 2-D images -> treat as (D,H,W,C) with free Z stride
                     We usually don't stride along D across files when extracting within a block,
                     so use (1, r, r) / (1, s, s).
    """
    if kind == "3d_single":
        return (roi_size, roi_size, roi_size), (roi_stride, roi_stride, roi_stride)
    # default: 2-D like
    return (1, roi_size, roi_size), (1, roi_stride, roi_stride)

def _infer_kind(path: Path, shape_dhwc: tuple[int,int,int,int]) -> str:
    D, H, W, C = shape_dhwc
    if path.is_dir():
        return "dir_2d"
    # single file: if D==1 but not from IMS/DHW → consider 2d_single; if D>1 → 3d_single
    return "3d_single" if D > 1 else "2d_single"

def image_to_feature_coord(img_coord: Tuple[int, int, int], *,
                           img_offset: Tuple[int, int, int],
                           roi_stride: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Map *img_coord* (z, y, x) to the corresponding feature volume index.

    Parameters
    ----------
    img_coord : tuple[int,int,int]
        Raw image coordinate.
    img_offset : tuple[int,int,int]
        Offset of the *first* processed voxel (often the beginning of the region).
    roi_stride : tuple[int,int,int]
        Stride of patch centres along each axis.
    """
    return tuple(((c - o) // s) for c, o, s in zip(img_coord, img_offset, roi_stride))

# -----------------------------------------------------------------------------
#                           F e a t u r e   e x t r a c t o r
# -----------------------------------------------------------------------------


# ────────────────────────────────────────────────────────── helpers
def _lookup(module: nn.Module, attr_path: str) -> nn.Module:
    tgt = module
    for attr in attr_path.split("."):
        tgt = getattr(tgt, attr) if attr else tgt
    return tgt


def _register_hook(layer: nn.Module, buffer: dict[str, torch.Tensor]):
    def hook(_, __, out):
        buffer["feat"] = out.detach()

    return layer.register_forward_hook(hook)


# ───────────────────────────────────────────────── extract
# ------------------------------ updated extract_features_to_zarr ------------------------------
def extract_features_to_zarr(
    *,
    vol_path: Union[str, Path],
    channel: int = 0,          # kept for IMS (selected channel index)
    model: nn.Module,
    zarr_path: Union[str, Path],
    global_offset: Tuple[int,int,int]=(0,0,0),
    whole_volume_size=None,    # optional override (D,H,W)
    region_size: Tuple[int,int,int],
    roi_size: int,             # <—— single int
    roi_stride: int,           # <—— single int
    batch_size: int = 256,
    device: str = "cuda",
    layer_path: str = "",
    pool_size: int | None = None,
    img_channel: int = 1,      # <—— NEW: desired per-slice channel count (1 or 3)
) -> None:
    model.eval().to(device)
    # hook
    layer = model if layer_path == "" else _lookup(model, layer_path)
    activ: dict[str, torch.Tensor] = {}
    handle = _register_hook(layer, activ)

    # open volume (normalized (D,H,W,C))
    with VolumeReader(vol_path, img_channel=img_channel, ims_channel=channel) as volume:
        D, H, W, C = volume.shape
        kind = _infer_kind(Path(vol_path), (D,H,W,C))
        roi3, stride3 = _roi_tuple_from_ints(kind, roi_size, roi_stride)

        # probe feature dim using a dummy patch
        with torch.no_grad():
            # dummy (B,C,D,H,W)
            dummy = torch.zeros(1, C, *roi3, device=device)
            if dummy.shape[2] == 1:
                dummy = dummy.squeeze(2)  # -> (1,C,H,W)
            _ = model(dummy)
            feat_sample = activ["feat"]
            if feat_sample.ndim == 5:
                adaptive_pool = nn.AdaptiveAvgPool3d((1,1,1)).to(device)
            elif feat_sample.ndim == 4:
                adaptive_pool = nn.AdaptiveAvgPool2d((1,1)).to(device)
            else:
                adaptive_pool = None
            if adaptive_pool is not None:
                feat_sample = adaptive_pool(feat_sample)
            feat_dim = int(feat_sample.flatten(1).shape[1])

        # region/stride bookkeeping (unchanged logic)
        step = [int(2 * (1 / 2) * r_size / r_stride - 1) for r_size, r_stride in zip(roi3, stride3)]
        margin = [int(s * s_size) for s, s_size in zip(step, stride3)]
        region_stride = [int(r_size - m) for r_size, m in zip(region_size, margin)]

        if whole_volume_size:
            d, h, w = whole_volume_size
        else:
            d, h, w = (D, H, W)

        num_blocks = [
            math.ceil((d - region_size[0]) / region_stride[0]) + 1,
            math.ceil((h - region_size[1]) / region_stride[1]) + 1,
            math.ceil((w - region_size[2]) / region_stride[2]) + 1,
        ]

        chunk_shape = [
            math.floor((region_size[0] - roi3[0]) / stride3[0]) + 1,
            math.floor((region_size[1] - roi3[1]) / stride3[1]) + 1,
            math.floor((region_size[2] - roi3[2]) / stride3[2]) + 1,
        ]

        zarr_shape = tuple(nb * cs for nb, cs in zip(num_blocks, chunk_shape)) + (feat_dim,)
        zarr_chunk = tuple(chunk_shape) + (feat_dim,)
        print(f"{region_stride =}, {zarr_shape= }, {zarr_chunk= }, {num_blocks= }")

        Path(zarr_path).parent.mkdir(parents=True, exist_ok=True)
        store = zarr.open(str(zarr_path), mode="w", shape=zarr_shape, dtype="float32", chunks=zarr_chunk)

        pbar = tqdm(total=math.prod(num_blocks), unit="block", desc="Feature extraction")

        for bz in range(num_blocks[0]):
            for by in range(num_blocks[1]):
                for bx in range(num_blocks[2]):
                    offset = (
                        bz * region_stride[0] + global_offset[0],
                        by * region_stride[1] + global_offset[1],
                        bx * region_stride[2] + global_offset[2],
                    )
                    # (d,h,w,c)
                    block = volume.read_block(offset=offset, size=region_size)

                    # pad to region_size spatially (channels already normalized)
                    pad = [max(0, region_size[i] - block.shape[i]) for i in range(3)]
                    if any(pad):
                        block = np.pad(
                            block,
                            [(0,pad[0]), (0,pad[1]), (0,pad[2]), (0,0)],
                            mode="constant"
                        )

                    # dataset expects (D,H,W,C)
                    dataset = SlidingWindowND(block, window=roi3, stride=stride3)
                    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

                    feats_all = []
                    with torch.no_grad():
                        for patches in loader:
                            # patches: (B,C,D,H,W)
                            patches = patches.to(device)

                            # If your model is 2-D (expects BCHW), uncomment:
                            if patches.shape[2] == 1:
                                patches = patches.squeeze(2)  # -> (B,C,H,W)

                            _ = model(patches)          # fills activ["feat"]
                            feat = activ["feat"]
                            if adaptive_pool is not None:
                                feat = adaptive_pool(feat)
                            feats_all.append(feat.flatten(1).cpu())  # [B, feat_dim]

                    feats_block = torch.cat(feats_all, dim=0).numpy().reshape(*chunk_shape, feat_dim)

                    store[
                        slice(bz * chunk_shape[0], (bz + 1) * chunk_shape[0]),
                        slice(by * chunk_shape[1], (by + 1) * chunk_shape[1]),
                        slice(bx * chunk_shape[2], (bx + 1) * chunk_shape[2]),
                        :
                    ] = feats_block.astype("float32")

                    pbar.update(1)

        pbar.close()
    handle.remove()
    print(f"Finished – features saved to {zarr_path}")

# -----------------------------------------------------------------------------
#                               e n t r y   p o i n t
# -----------------------------------------------------------------------------
    
#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from lib.arch.ae import build_encoder_model, load_encoder2encoder 


#%%

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', default='config/pipeline.yaml', help='Path to pipeline.yaml')
parser.add_argument('-view_zarr', action='store_true', help='Only summarise and plot the feature Zarr; skip extraction')
parser.add_argument('-n', type=int, default=8, help='Number of slices per axis to plot/save')
args = parser.parse_args()

from pipeline import load_cfg 
from lib.utils.yaml_utils import  to_attr
# from config.load_config import load_cfg
cfg = to_attr(load_cfg(args.cfg))
vol_path = cfg._run.input_image


# Save features under <output_root>/<run_id>/data/<zarr_name>
save_zarr_path = str(Path(cfg.paths.output_root) / cfg.run_id / 'data' / cfg.paths.zarr_name)

# Params used for extraction and logging
paras = cfg.feature_extract

if not args.view_zarr:
    model = build_encoder_model(cfg.autoencoder, dims=cfg.dims)
    print(f"{model= }")
    _ = torch.load(cfg.paths.ae_weight_path)
    load_encoder2encoder(model, cfg.paths.ae_weight_path)

    extract_features_to_zarr(
        vol_path= vol_path,
        channel=cfg.paths.channel,
        model=model,
        zarr_path=save_zarr_path,
        global_offset= paras.global_offset,
        whole_volume_size=paras.whole_volume_size,
        region_size=paras.region_size,
        roi_size=paras.roi_size,
        roi_stride=paras.roi_stride,
        batch_size=paras.batch_size,
        device="cuda",
        layer_path="",  # pick *one* internal layer
        img_channel=int(cfg.paths.img_channel),  # <—— NEW
        # Adaptive pooling is applied automatically inside extract_features_to_zarr
    )


#%%
import matplotlib.pyplot as plt
from pathlib import Path
import zarr
import numpy as np
# -----------------------------------------------------------------------------
#                       q u i c k   v a l i d a t i o n   u t i l s
# -----------------------------------------------------------------------------

def summarise_zarr(path: str | Path):
    """Print and return basic metadata (shape, chunks, dtype)."""
    arr = zarr.open(str(path), mode="r")
    summary = f"Shape  : {arr.shape}\nChunks : {arr.chunks}\nDType  : {arr.dtype}\n"
    print(summary)
    return summary


def plot_zarr_slices(path: str | Path, n: int = 6, *, pca_rgb: bool = False, channel_axis: int = -1, save_dir: str | Path | None = None, prefix: str = "extract"):
    """Maximum-projection slices along Z/Y/X for a quick sanity check.

    Parameters
    ----------
    path : str | Path
        Zarr array path.
    n : int, default 6
        How many slices per axis to show.
    pca_rgb : bool, default False
        If *True*, also show an RGB PCA view of the middle Z-slice.
    channel_axis : int, default -1
        Channel axis location. Use -1 for channel-last (D,H,W,C),
        or 0 for channel-first (C,D,H,W).
    """
    if channel_axis not in (-1, 0):
        raise ValueError("channel_axis must be -1 (C-last) or 0 (C-first).")

    arr = zarr.open(str(path), mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D array, got shape {arr.shape}.")

    if channel_axis == -1:
        D, H, W, C = arr.shape
        z_img = lambda z: arr[z, :, :, :].max(-1)
        y_img = lambda y: arr[:, y, :, :].max(-1)
        x_img = lambda x: arr[:, :, x, :].max(-1).T
        get_mid_slice_for_pca = lambda mid_z: arr[mid_z, :, :, :]  # (H, W, C)
    else:  # channel_axis == 0  -> (C, D, H, W)
        C, D, H, W = arr.shape
        z_img = lambda z: arr[:, z, :, :].max(0)
        y_img = lambda y: arr[:, :, y, :].max(0)
        x_img = lambda x: arr[:, :, :, x].max(0).T
        # Convert (C, H, W) -> (H, W, C) for PCA
        get_mid_slice_for_pca = lambda mid_z: np.moveaxis(arr[:, mid_z, :, :], 0, -1)

    z_lin = np.linspace(0, D - 1, n, dtype=int)
    y_lin = np.linspace(0, H - 1, n, dtype=int)
    x_lin = np.linspace(0, W - 1, n, dtype=int)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    for i, z in enumerate(z_lin):
        axes[0, i].imshow(z_img(z), cmap="gray")
        axes[0, i].set_title(f"Z {z}")
        axes[0, i].axis("off")
    for i, y in enumerate(y_lin):
        axes[1, i].imshow(y_img(y), cmap="gray")
        axes[1, i].set_title(f"Y {y}")
        axes[1, i].axis("off")
    for i, x in enumerate(x_lin):
        axes[2, i].imshow(x_img(x), cmap="gray")
        axes[2, i].set_title(f"X {x}")
        axes[2, i].axis("off")

    plt.tight_layout()
    saved_paths = []
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        mp_path = save_dir / f"{prefix}_maxproj.png"
        fig.savefig(mp_path, dpi=150)
        saved_paths.append(str(mp_path))
        plt.close(fig)

    if pca_rgb:
        try:
            from sklearn.decomposition import PCA
            mid_z = D // 2
            mid = get_mid_slice_for_pca(mid_z)   # (H, W, C)
            Hm, Wm, Cm = mid.shape
            if Cm < 3:
                print(f"PCA-RGB needs >=3 channels, but got C={Cm}. Skipping.")
            else:
                flat = mid.reshape(-1, Cm).astype(np.float32)
                pca = PCA(n_components=3).fit_transform(flat)
                rgb = (pca - pca.min(0)) / (pca.ptp(0) + 1e-7)
                rgb = rgb.reshape(Hm, Wm, 3)
                fig2 = plt.figure(figsize=(6, 6))
                plt.title("Mid-Z PCA-RGB")
                plt.imshow(rgb)
                if save_dir is not None:
                    pca_path = Path(save_dir) / f"{prefix}_pca_rgb.png"
                    fig2.savefig(pca_path, dpi=150)
                    saved_paths.append(str(pca_path))
                    plt.close(fig2)
        except ImportError:
            print("Install scikit-learn for PCA-RGB view → `pip install scikit-learn`.\n")
    # Only show if no save_dir specified
    if save_dir is None:
        plt.show()
    return saved_paths

# Convenience: view-only wrapper that saves summary + images
def view_zarr_feats(zarr_path: str | Path, out_dir: str | Path, *, n: int = 8, pca_rgb: bool = True, channel_axis: int = -1, prefix: str = "features"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_text = summarise_zarr(zarr_path)
    (out_dir / "zarr_summary.txt").write_text(summary_text)
    saved_imgs = plot_zarr_slices(zarr_path, n=n, pca_rgb=pca_rgb, channel_axis=channel_axis, save_dir=out_dir, prefix=prefix)
    return {"zarr_path": str(zarr_path), "saved_images": saved_imgs, "summary_file": str(out_dir / "zarr_summary.txt")}

#%%
# Prepare extraction output folder for logs and configs
extract_dir = Path(cfg.paths.output_root) / cfg.run_id / cfg.paths.extract_out_folder
extract_dir.mkdir(parents=True, exist_ok=True)

# Save summary and plots
view_info = view_zarr_feats(save_zarr_path, extract_dir, n=args.n, pca_rgb=True, channel_axis=-1, prefix="features")
saved_imgs = view_info["saved_images"]

# Persist extraction arguments for reproducibility (only in extraction mode)
if not args.view_zarr:
    extract_args = {
        "vol_path": str(vol_path),
        "channel": int(cfg.paths.channel),
        "zarr_path": save_zarr_path,
        "global_offset": list(paras.global_offset) if paras.global_offset is not None else None,
        "whole_volume_size": list(paras.whole_volume_size) if paras.whole_volume_size is not None else None,
        "region_size": list(paras.region_size),
        "roi_size": int(paras.roi_size),
        "roi_stride": int(paras.roi_stride),
        "batch_size": int(paras.batch_size),
        "device": "cuda",
        "layer_path": "",
        "dims": int(cfg.dims),
        "model_class": model.__class__.__name__,
        "saved_images": saved_imgs,
    }
    (extract_dir / "extract_args.json").write_text(json.dumps(extract_args, indent=2))
