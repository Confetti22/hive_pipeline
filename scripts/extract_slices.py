#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import tifffile as tiff
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 2D z-slices from 3D volumes and save as TIFF images."
    )
    parser.add_argument("input_dir", type=str, help="Directory containing 3D volumes (.tif/.tiff or .npy)")
    parser.add_argument("output_dir", type=str, help="Directory to write extracted 2D TIFF slices")
    return parser.parse_args()


def is_volume_file(path: Path) -> bool:
    ext = path.suffix.lower()
    return ext in {".tif", ".tiff", ".npy"}


def iter_volume_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if p.is_file() and is_volume_file(p):
            yield p


def load_volume(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        vol = tiff.imread(str(path))
    elif path.suffix.lower() == ".npy":
        vol = np.load(str(path))
    else:
        raise ValueError(f"Unsupported file type: {path}")

    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape} for {path}")

    # Ensure shape is (Z, H, W). Common alternates: (H, W, Z)
    z, y, x = vol.shape
    if z not in (16, 32, 64, 128) and vol.shape[-1] in (16, 32, 64, 128):
        vol = np.moveaxis(vol, -1, 0)

    return vol


def save_slice(slice2d: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write the slice exactly as-is without any preprocessing
    tiff.imwrite(str(out_path), slice2d)


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Input directory not found: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    vol_files = list(iter_volume_files(in_dir))
    if not vol_files:
        raise SystemExit(f"No volume files (.tif/.tiff/.npy) found in {in_dir}")

    # Calculate total number of slices for progress bar
    total_slices = 0
    for vol_path in vol_files:
        vol = load_volume(vol_path)
        total_slices += vol.shape[0]

    # Process volumes with progress bar
    processed_slices = 0
    with tqdm(total=total_slices, desc="Extracting slices", unit="slice") as pbar:
        for vol_path in vol_files:
            vol = load_volume(vol_path)
            if vol.shape[1:] != (512, 512) or vol.shape[0] != 32:
                # Allow other shapes but warn; still proceed
                pass

            z_slices = vol.shape[0]
            base = vol_path.stem
            for z in range(z_slices):
                out_name = f"{base}_z{z:03d}.tiff"
                save_slice(vol[z], out_dir / out_name)
                processed_slices += 1
                pbar.update(1)

    print(f"Done. Wrote {processed_slices} slices to {out_dir}")


if __name__ == "__main__":
    main()


