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
 # -----------------------------------------------------------------------------
#                       I / O   a n d   v o l u m e   r e a d e r
# -----------------------------------------------------------------------------
from pathlib import Path
import re
from typing import Tuple
import numpy as np
from tifffile import memmap as tiff_memmap

class VolumeReader:
    """Handle large 3-D volumes from:
       1) single `.ims`
       2) single `.tif/.tiff` (2-D or multi-page 3-D)
       3) directory of 2-D `.tif/.tiff` files → stacked along Z

    Unified API:
        with VolumeReader(path, channel=0) as vol:
            patch = vol.read_block(offset=(z,y,x), size=(d,h,w))
    """

    def __init__(self, path: str | Path, channel: int = 0):
        self.path = Path(path)
        self.channel = channel
        self._handle = None          # Ims_Image or memmapped ndarray for single file
        self._is_dir = self.path.is_dir()
        self._dir_files: list[Path] = []
        self._shape: Tuple[int, int, int] | None = None

    # ------------------------------ helpers
    @staticmethod
    def _natural_key(p: Path):
        # Sort “slice_2.tif” before “slice_10.tif”
        parts = re.split(r'(\d+)', p.name)
        return [int(s) if s.isdigit() else s.lower() for s in parts]

    # ------------------------------------------------------------------ context
    def __enter__(self):
        if self._is_dir:
            # Collect 2-D TIFFs and sort naturally
            self._dir_files = sorted(
                [p for p in self.path.iterdir() if p.suffix.lower() in {'.tif', '.tiff'}],
                key=self._natural_key
            )
            if not self._dir_files:
                raise ValueError(f"No .tif/.tiff files found in directory: {self.path}")

            # Probe H,W from the first file
            first = tiff_memmap(self._dir_files[0])
            if first.ndim != 2:
                raise ValueError("Directory mode expects each TIFF to be 2-D (H,W).")
            H, W = map(int, first.shape)
            D = len(self._dir_files)
            self._shape = (D, H, W)
            return self

        # File path: .ims or .tif/.tiff
        suffix = self.path.suffix.lower()
        if suffix == ".ims":
            from helper.image_reader import Ims_Image  # local import to avoid heavy deps
            self._handle = Ims_Image(str(self.path), channel=self.channel)
            # `rois[0]` → (z,y,x,d,h,w); store full size (d,h,w)
            self._shape = tuple(int(x) for x in self._handle.rois[0][3:])
        elif suffix in {".tif", ".tiff"}:
            arr = tiff_memmap(self.path)  # can be 2-D or 3-D (pages, H, W)
            if arr.ndim == 2:
                # Promote to (1,H,W)
                self._handle = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                self._handle = arr
            else:
                raise ValueError(f"Unsupported TIFF ndim={arr.ndim}, expected 2 or 3.")
            self._shape = tuple(int(x) for x in self._handle.shape)  # (D,H,W)
        else:
            raise ValueError(f"Unsupported volume format: {self.path.suffix}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close IMS handle if present
        if self._handle is not None and hasattr(self._handle, "close"):
            self._handle.close()

    # --------------------------------------------------------------------- meta
    @property
    def shape(self) -> Tuple[int, int, int]:
        if self._shape is None:
            raise RuntimeError("VolumeReader not opened. Use as a context manager.")
        return self._shape

    # ------------------------------------------------------------- random block
    def read_block(self, *, offset: Tuple[int, int, int], size: Tuple[int, int, int]) -> np.ndarray:
        """Read a 3-D sub-volume starting at *offset* with *size* (z-first)."""
        z0, y0, x0 = offset
        dz, dh, dw = size

        if self._is_dir:
            D, H, W = self.shape
            z1 = min(z0 + dz, D)
            if z0 >= D:
                # Entirely out of bounds on Z → return empty (caller pads later)
                return np.zeros((0, min(dh, H - y0), min(dw, W - x0)), dtype=np.float32)

            # Gather the needed 2-D slices and crop Y,X
            slices = []
            for zi in range(z0, z1):
                arr2d = tiff_memmap(self._dir_files[zi])  # (H,W)
                patch2d = arr2d[y0:y0 + dh, x0:x0 + dw]
                slices.append(np.asarray(patch2d))
            if not slices:
                return np.zeros((0, dh, dw), dtype=np.float32)
            return np.stack(slices, axis=0)  # (dz_eff, dh_eff, dw_eff)

        # Single file: IMS or TIFF memmap
        if hasattr(self._handle, "from_roi"):  # IMS path
            coords = np.array([z0, y0, x0, dz, dh, dw], dtype=np.int64)
            return self._handle.from_roi(coords=coords, level=0)

        # TIFF path: ndarray with shape (D,H,W)
        return np.asarray(self._handle[z0:z0 + dz, y0:y0 + dh, x0:x0 + dw])

# -----------------------------------------------------------------------------
#                  C o o r d i n a t e   m a p p i n g   h e l p e r
# -----------------------------------------------------------------------------

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
def extract_features_to_zarr(
    *,
    vol_path: Union[str, Path],
    channel:int = 0, #channel for ims image
    model: nn.Module,
    zarr_path: Union[str, Path],
    global_offset: Tuple[int,int,int]= (0,0,0),
    whole_volume_size =None,
    region_size: Tuple[int, int, int],
    roi_size: int,
    roi_stride: int,
    batch_size: int = 256,
    device: str = "cuda",
    # NEW ↓
    layer_path: str = "",           # path to layer inside the model (“” = model output)
    pool_size: int | None = None,   # deprecated: manual avg pool size (ignored)
) -> None:
    """Extract feature map into shape (D, H, W, C).

    Notes:
    - Always applies adaptive average pooling after the target layer/model to
      collapse spatial dimensions to 1 (3D → (1,1,1); 2D → (1,1)). This removes
      the need to manually tune an avg_pool kernel size and guarantees a fixed
      feature length per ROI sample.
    """
    model.eval().to(device)

    # Hook the requested layer
    layer = model if layer_path == "" else _lookup(model, layer_path)
    activ: dict[str, torch.Tensor] = {}
    handle = _register_hook(layer, activ)

    # ---------------------------------------------------------------- size probe
    with torch.no_grad():
        dummy = torch.zeros(1, 1, *roi_size, device=device)
        _ = model(dummy)
        feat_sample = activ["feat"]

        # Always apply adaptive pooling to collapse spatial dims deterministically
        if feat_sample.ndim == 5:      # (B, C, D, H, W)
            adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1)).to(device)
        elif feat_sample.ndim == 4:    # (B, C, H, W)
            adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        else:
            adaptive_pool = None  # already flat or unexpected; skip pooling

        if adaptive_pool is not None:
            feat_sample = adaptive_pool(feat_sample)
        feat_dim = feat_sample.numel()

    # ───────────────────────────── Zarr grid bookkeeping (unchanged logic)
    step = [int(2 * (1 / 2) * r_size / r_stride - 1) for r_size, r_stride in zip(roi_size, roi_stride)]
    margin = [int(s * s_size) for s, s_size in zip(step, roi_stride)]
    region_stride = [int(r_size - m) for r_size, m in zip(region_size, margin)]

    with VolumeReader(vol_path,channel=channel) as volume:
        if whole_volume_size:
            d, h, w = whole_volume_size
        else:
            d, h, w = volume.shape
        num_blocks = [
            math.ceil((d - region_size[0]) / region_stride[0]) + 1,
            math.ceil((h - region_size[1]) / region_stride[1]) + 1,
            math.ceil((w - region_size[2]) / region_stride[2]) + 1,
        ]

        chunk_shape = [
            math.floor((region_size[0] - roi_size[0]) / roi_stride[0]) + 1,
            math.floor((region_size[1] - roi_size[1]) / roi_stride[1]) + 1,
            math.floor((region_size[2] - roi_size[2]) / roi_stride[2]) + 1,
        ]

        zarr_shape = tuple(nb * cs for nb, cs in zip(num_blocks, chunk_shape)) + (feat_dim,)
        zarr_chunk = tuple(chunk_shape) + (feat_dim,)
        print(f"{region_stride =}, {zarr_shape= }, {zarr_chunk= },{num_blocks= }")
        # Ensure parent directory for Zarr path exists
        Path(zarr_path).parent.mkdir(parents=True, exist_ok=True)
        store = zarr.open(str(zarr_path), mode="w", shape=zarr_shape, dtype="float32", chunks=zarr_chunk)

        pbar = tqdm(total=math.prod(num_blocks), unit="block", desc="Feature extraction")

        # --------------------------------------------------- iterate volume grid
        for bz in range(num_blocks[0]):
            for by in range(num_blocks[1]):
                for bx in range(num_blocks[2]):
                    offset = (
                        bz * region_stride[0] + global_offset[0],
                        by * region_stride[1] + global_offset[1],
                        bx * region_stride[2] + global_offset[2],
                    )
                    block = volume.read_block(offset=offset, size=region_size)

                    pad = [max(0, region_size[i] - block.shape[i]) for i in range(3)]
                    if any(pad):
                        block = np.pad(block, [(0, pad[0]), (0, pad[1]), (0, pad[2])], mode="constant")

                    dataset = SlidingWindowND(block, window=roi_size, stride=roi_stride)
                    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

                    feats_all = []
                    with torch.no_grad():
                        for patches in loader:
                            patches = patches.to(device)
                            _ = model(patches)            # fills activ["feat"]
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
        "roi_size": list(paras.roi_size),
        "roi_stride": list(paras.roi_stride),
        "batch_size": int(paras.batch_size),
        "device": "cuda",
        "layer_path": "",
        "dims": int(cfg.dims),
        "model_class": model.__class__.__name__,
        "saved_images": saved_imgs,
    }
    (extract_dir / "extract_args.json").write_text(json.dumps(extract_args, indent=2))
