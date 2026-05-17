from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import h5py
import numpy as np
import torch

from lib.arch.segmodel import build_dpt
from lib.inferencers.tilled_inference2d3d import eval_full_roi


DEFAULT_INPUT = "../../e5_data/t1779/fourview/22.h5"
DEFAULT_OUTPUT = "../../e5_data/t1779/fourview/dino_feats/8um_z_stack_feats.zarr"


def _slice_len(s: slice, source_size: int) -> int:
    return len(range(*s.indices(source_size)))


def _map_slice(local_slice: slice, base_slice: slice, source_size: int) -> slice:
    base_start, _, base_step = base_slice.indices(source_size)
    local_start, local_stop, local_step = local_slice.indices(_slice_len(base_slice, source_size))
    return slice(
        base_start + local_start * base_step,
        base_start + local_stop * base_step,
        base_step * local_step,
    )


class H5StridedVolume:
    """Lazy (D,H,W) view over one channel of an HDF5 [C,Z,Y,X] dataset."""

    def __init__(
        self,
        dataset: h5py.Dataset,
        channel: int,
        source_slices: Sequence[slice],
    ):
        self.dataset = dataset
        self.channel = channel
        self.source_slices = tuple(source_slices)
        self.shape = tuple(
            _slice_len(s, size)
            for s, size in zip(self.source_slices, self.dataset.shape[1:])
        )

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) != 3:
            raise IndexError(f"Expected 3D index, got {index}")

        mapped = []
        for item, source_slice, source_size in zip(index, self.source_slices, self.dataset.shape[1:]):
            if isinstance(item, int):
                base_start, _, base_step = source_slice.indices(source_size)
                mapped.append(base_start + item * base_step)
            elif isinstance(item, slice):
                mapped.append(_map_slice(item, source_slice, source_size))
            else:
                raise TypeError(f"Unsupported index component: {item!r}")
        return self.dataset[(self.channel, *mapped)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DINO/DPT feature maps from four-view HDF5 reconstruction data."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="HDF5 file containing [C,Z,Y,X] data.")
    parser.add_argument("--dataset", default="data", help="HDF5 dataset key.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output Zarr path for feature maps.")

    #raw image input range and resolution 
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--z-start", type=int, default=180)
    parser.add_argument("--z-stop", type=int, default=420)
    parser.add_argument("--z-step", type=int, default=8)
    parser.add_argument("--yx-step", type=int, default=8)
    
    #tiled feature inference setttings
    parser.add_argument("--tile", type=int, nargs=3, default=(1, 512, 512), metavar=("D", "H", "W"))
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--feature-up-scale-factor", type=int, default=1)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    tile = tuple(args.tile)
    if args.feature_up_scale_factor == 1 and (tile[1] % 16 != 0 or tile[2] % 16 != 0):
        raise ValueError("When feature_up_scale_factor=1, tile H/W must be divisible by 16.")
    if not 0 <= args.overlap < 1:
        raise ValueError("overlap must be in [0, 1).")


def build_feature_model(args: argparse.Namespace):
    segmodel = build_dpt(
        dims=3,
        n_classes=args.n_classes,
        model_dir=args.model_dir,
        linear_prob=False,
        feature_up_scale_factor=args.feature_up_scale_factor,
    )
    segmodel.seg_model.eval().to(args.device)
    if hasattr(segmodel.seg_model, "set_feature_only"):
        segmodel.seg_model.set_feature_only(True)
    return segmodel


def run_feature_extraction(args: argparse.Namespace):
    validate_args(args)
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z_slice = slice(args.z_start, args.z_stop, args.z_step)
    y_slice = slice(None, None, args.yx_step)
    x_slice = slice(None, None, args.yx_step)

    with h5py.File(input_path, "r") as h5_file:
        dataset = h5_file[args.dataset]
        volume = H5StridedVolume(dataset, args.channel, (z_slice, y_slice, x_slice))
        print(f"Input dataset: {input_path}:{args.dataset} shape={dataset.shape} dtype={dataset.dtype}")
        print(f"Lazy trial volume shape: {volume.shape}")

        segmodel = build_feature_model(args)
        _, feats = eval_full_roi(
            segmodel,
            volume,
            device=args.device,
            tile=tuple(args.tile),
            capture_features=True,
            tv_denoise_weight=0.0,
            overlap=args.overlap,
            collect_prediction=False,
            feature_output_path=str(output_path),
        )

    print(f"Saved feature map: {output_path}")
    print(f"Feature zarr shape: {None if feats is None else feats.shape}")


if __name__ == "__main__":
    run_feature_extraction(parse_args())
