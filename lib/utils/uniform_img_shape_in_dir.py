#!/usr/bin/env python3
"""
Pad all TIFF files in a directory to the same shape (the maximum shape across the set),
then save to an output directory with the same filenames.

Defaults:
- Alignment: top-left (pads only on the "end" of each axis).
- Fill value: 0
- Matches *.tif and *.tiff (configurable)
- Preserves each file's dtype
- Writes BigTIFF automatically when size > 4 GiB

Usage:
  python pad_tiffs.py /path/to/input_dir /path/to/output_dir \
      --align top-left --fill 0 --pattern "*.tif,*.tiff"

Notes:
- Works for N-D arrays (2D images or 3D/ND stacks).
- For center alignment, padding is split before/after each axis.
- Requires: numpy, tifffile
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Sequence, Tuple
import numpy as np
import tifffile as tif
import math
import sys

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pad all TIFFs in a folder to the same (max) shape.")
    p.add_argument("input_dir", type=Path, default='/home/confetti/e5_data/wide_filed/nuclei_channel',help="Directory containing input .tif/.tiff files")
    p.add_argument("output_dir", type=Path, default='/home/confetti/e5_data/wide_filed/padded_nuclei_channel',help="Directory to write padded TIFFs")
    p.add_argument("--pattern", type=str, default="*.tif,*.tiff",
                   help="Comma-separated glob(s) for TIFF files (default: *.tif,*.tiff)")
    p.add_argument("--align", choices=["top-left", "center"], default="top-left",
                   help="Where to keep the original content; default: top-left")
    p.add_argument("--fill", type=float, default=0.0,
                   help="Padding fill value (cast to each image dtype). Default 0")
    p.add_argument("--bigtiff-threshold", type=int, default=(2**32 - 1),
                   help="Bytes threshold to switch on BigTIFF (default ~4 GiB).")
    p.add_argument("--verbose", action="store_true", help="Print extra info.")
    return p.parse_args()

def gather_files(input_dir: Path, patterns: str) -> List[Path]:
    exts = [pat.strip() for pat in patterns.split(",") if pat.strip()]
    files: List[Path] = []
    for pat in exts:
        files.extend(sorted(input_dir.rglob(pat)))
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq

def compute_max_shape(files: Sequence[Path], verbose: bool=False) -> Tuple[int, ...]:
    max_shape = None
    for f in files:
        try:
            arr = tif.imread(f)
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: failed to read ({e})", file=sys.stderr)
            continue
        if verbose:
            print(f"[INFO] {f.name} shape={arr.shape}, dtype={arr.dtype}")
        if max_shape is None:
            max_shape = arr.shape
        else:
            if arr.ndim != len(max_shape):
                # If you need to support mixing 2D and 3D, you can expand dims here.
                raise ValueError(f"Mixed dimensionalities detected: {f.name} has ndim={arr.ndim}, "
                                 f"previous max ndim={len(max_shape)}. Normalize your data first.")
            max_shape = tuple(max(s, m) for s, m in zip(arr.shape, max_shape))
    if max_shape is None:
        raise RuntimeError("No readable TIFF files found.")
    return max_shape

def compute_pad_width(src_shape: Tuple[int, ...], tgt_shape: Tuple[int, ...], align: str) -> List[Tuple[int, int]]:
    if len(src_shape) != len(tgt_shape):
        raise ValueError("src_shape and tgt_shape must have same dimensionality.")
    pads: List[Tuple[int, int]] = []
    for s, t in zip(src_shape, tgt_shape):
        if s > t:
            raise ValueError(f"Source dimension {s} larger than target {t}. (Unexpected since target is max.)")
        gap = t - s
        if align == "top-left":
            pads.append((0, gap))
        elif align == "center":
            before = gap // 2
            after = gap - before
            pads.append((before, after))
        else:
            raise ValueError(f"Unknown align={align}")
    return pads

def pad_array(arr: np.ndarray, tgt_shape: Tuple[int, ...], align: str, fill_value: float) -> np.ndarray:
    pads = compute_pad_width(arr.shape, tgt_shape, align)
    # Choose constant mode padding with fill_value (cast later)
    # np.pad requires pad_width per axis
    padded = np.pad(arr, pads, mode="constant", constant_values=fill_value).astype(arr.dtype, copy=False)
    return padded

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def write_tiff(path: Path, data: np.ndarray, bigtiff_threshold: int) -> None:
    ensure_parent_dir(path)
    need_bigtiff = data.nbytes > bigtiff_threshold
    # Save. If data is multi-page/ND, tifffile will write it as such.
    tif.imwrite(
        path,
        data,
        bigtiff=need_bigtiff,
        photometric="minisblack" if (data.ndim == 2 or (data.ndim > 2 and data.shape[-1] != 3)) else None,
    )

def main():
    args = parse_args()
    in_dir: Path = args.input_dir
    out_dir: Path = args.output_dir

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"[ERR] Input directory does not exist or is not a directory: {in_dir}")

    files = gather_files(in_dir, args.pattern)
    if not files:
        raise SystemExit(f"[ERR] No TIFF files match patterns '{args.pattern}' in {in_dir}")

    if args.verbose:
        print(f"[INFO] Found {len(files)} files. Computing max shape...")

    max_shape = compute_max_shape(files, verbose=args.verbose)
    if args.verbose:
        print(f"[INFO] Target (max) shape: {max_shape}")

    count_ok = 0
    for f in files:
        rel = f.relative_to(in_dir)
        out_path = out_dir / rel
        try:
            arr = tif.imread(f)
            padded = pad_array(arr, max_shape, args.align, args.fill)
            # Cast fill to dtype was handled in pad_array via astype(arr.dtype)
            write_tiff(out_path, padded, args.bigtiff_threshold)
            count_ok += 1
            if args.verbose:
                print(f"[OK] {f.name} -> {out_path} | shape {arr.shape} -> {padded.shape}")
        except Exception as e:
            print(f"[FAIL] {f}: {e}", file=sys.stderr)

    print(f"[DONE] Wrote {count_ok}/{len(files)} files to: {out_dir}")

if __name__ == "__main__":
    main()