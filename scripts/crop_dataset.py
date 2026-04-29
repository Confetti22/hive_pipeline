# ----------------------------------
# scripts/crop_rois.py (Step 1: crop)
# ----------------------------------
# Reads:   paths.input_image, paths.level, paths.channel, crop.*
# Writes:  runs/<run_id>/data/<ae_train_folder>/*.tif and .../<ae_test_folder>/*.tif

import argparse
from pathlib import Path
import numpy as np
import tifffile as tif
import math
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
# Optional IMS helper
from helper.image_reader import Ims_Image


"""
1.7 for 1um mouse brain nissl image
2.7 for 4 um macaque brain nissl image
5 for wide_field_nuclei image

"""

# ----------------------------- utils -----------------------------
def shannon_entropy(img):
    vals, counts = np.unique(img, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def entropy_filter(l_thres=1.4, h_thres=100.0,v=False):
    def _f(img):
        ent = shannon_entropy(img)
        if v and ent>= l_thres:
            print(f"[entropy] {ent:.4f}")
        return (ent >= l_thres) and (ent <= h_thres)
    return _f


def load_cfg(p):
    import yaml
    with open(p, "r") as f:
        return yaml.safe_load(f)


def parse_roi_size(roi_cfg, spatial_dims: int):
    if isinstance(roi_cfg, (int, float)):
        return (int(roi_cfg),) * spatial_dims
    if isinstance(roi_cfg, (list, tuple)) and len(roi_cfg) == spatial_dims:
        return tuple(int(v) for v in roi_cfg)
    raise SystemExit(f"[ERR] roi_size must be int or list/tuple of len {spatial_dims}, got: {roi_cfg}")


def default_sample_range(shape):
    return [[0, int(s)] for s in shape]


def clamp_axis(axis, spatial_dims):
    return max(0, min(int(axis), spatial_dims - 1))


def split_ranges(sample_range, axis: int, train_ratio: float, v=True):
    sr = [list(r) for r in sample_range]
    lb, ub = sr[axis]
    cut = int(lb + train_ratio * (ub - lb))
    srt, srs = [list(map(tuple, sr)) for _ in (0, 1)]
    srt[axis] = (lb, cut)
    srs[axis] = (cut, ub)
    if v:
        print(f"[split] axis={axis} cut={cut}  train={srt[axis]}  test={srs[axis]}")
    # convert back to [[lb,ub], ...]
    srt = [list(r) for r in srt]
    srs = [list(r) for r in srs]
    return srt, srs


def distribute_counts(total: int, n_bins: int):
    base, rem = divmod(total, n_bins)
    return [base + (1 if i < rem else 0) for i in range(n_bins)]


# ------------------------ ROI samplers (2D/3D) ------------------------
def _rand_idx(sr_pair, size):
    a, b = sr_pair
    span = b - a
    if span <= size:
        return a
    return np.random.randint(a, b - size)


def sample_roi_2d(arr, sr, roi_size, filt, max_try=5000):
    H, W = roi_size
    for _ in range(max_try):
        y = _rand_idx(sr[0], H)
        x = _rand_idx(sr[1], W)
        roi = arr[y:y + H, x:x + W]
        if filt(roi): return roi
    return None


def sample_roi_3d(vol, sr, roi_size, filt, max_try=5000):
    D, H, W = roi_size
    for _ in range(max_try):
        z = _rand_idx(sr[0], D)
        y = _rand_idx(sr[1], H)
        x = _rand_idx(sr[2], W)
        roi = vol[z:z + D, y:y + H, x:x + W]
        if filt(roi): return roi
    return None


# --------------------------- write helpers ---------------------------
def write_set(get_roi, count, out_dir, name_fmt, progress_every=200):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    while saved < count:
        roi = get_roi()
        if roi is None:
            raise SystemExit("[ERR] Not enough ROIs passing entropy filter; relax thresholds/sizes.")
        tif.imwrite(str(out_dir / name_fmt(saved)), roi)
        saved += 1
        if saved % progress_every == 0 or saved == count:
            print(f"[crop] saved {saved}/{count} → {out_dir.name}")
    return saved


# ================================ main ================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-cfg", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    crop = cfg["crop"]
    paths = cfg["paths"]

    run_id = cfg["run_id"]
    root = Path(paths.get("output_root", "runs")).expanduser().absolute() / run_id
    data_root = root / "data"
    train_dir = data_root / paths["ae_train_folder"]
    test_dir  = data_root / paths["ae_test_folder"]

    img_path = Path(paths["input_image"])
    level = int(paths.get("level", 0))
    channel = int(paths.get("channel", 0))

    # configs with safe defaults for None/empty
    roi_cfg = crop.get("roi_size") if crop.get("roi_size") not in (None, "") else crop.get("rois_size")
    if roi_cfg in (None, ""):
        raise SystemExit("[ERR] crop.roi_size is required.")

    total = int(crop.get("amount") or 0)
    if total <= 0:
        raise SystemExit("[ERR] crop.amount must be > 0")

    train_ratio = float(crop.get("train_split") or 0.8)
    ent_th = float(crop.get("entropy_thres") or 1.4)
    split_axis_cfg = crop.get("split_axis", 0)
    sample_range_cfg = crop.get("sample_range", None)
    if isinstance(sample_range_cfg, str) and sample_range_cfg.lower() == "none":
        sample_range_cfg = None

    filt = entropy_filter(l_thres=ent_th)
    train_n = int(round(total * train_ratio))
    test_n  = total - train_n

    # -------------------------- .ims branch --------------------------
    if str(img_path).lower().endswith((".ims", ".ims.part")):
        if Ims_Image is None:
            raise SystemExit("[ERR] .ims input requires helper.image_reader.Ims_Image")
        ims = Ims_Image(str(img_path), channel=channel)
        shape = tuple(ims.info[level]["data_shape"])  # (Z,Y,X)
        spatial_dims = 3

        roi_size = parse_roi_size(roi_cfg, spatial_dims)
        split_axis = clamp_axis(split_axis_cfg, spatial_dims)

        if sample_range_cfg is None:
            sample_range = default_sample_range(shape)
        else:
            if len(sample_range_cfg) != spatial_dims:
                raise SystemExit(f"[ERR] sample_range dims != {spatial_dims}")
            sample_range = [list(map(int, r)) for r in sample_range_cfg]

        train_sr, test_sr = split_ranges(sample_range, split_axis, train_ratio)

        def make_get_roi(sr):
            def _f():
                roi, _ = ims.get_random_roi(
                    filter=filt, roi_size=roi_size, level=level,
                    skip_gap=False, sample_range=sr, margin=0
                )
                return roi
            return _f

        tr_saved = write_set(make_get_roi(train_sr), train_n, train_dir, lambda i: f"{i:05d}.tif")
        te_saved = write_set(make_get_roi(test_sr),  test_n,  test_dir,  lambda i: f"{i:05d}.tif", progress_every=100)
        print(f"[OK] crop done (.ims) → train:{tr_saved}  test:{te_saved}")
        return

    # -------------------- TIFF (file or directory) --------------------
       # -------------------- TIFF (file or directory) --------------------
    if img_path.is_dir():
        tiffs = sorted([p for p in img_path.iterdir() if p.suffix.lower() in (".tif", ".tiff")])
        if not tiffs:
            raise SystemExit(f"[ERR] No .tif/.tiff in directory: {img_path}")

        probe = tif.imread(str(tiffs[0]))
        if probe.ndim not in (2, 3):
            raise SystemExit(f"[ERR] Unsupported TIFF ndim {probe.ndim} in dir (expect 2 or 3)")
        spatial_dims = probe.ndim
        shape = probe.shape if spatial_dims == 2 else (probe.shape[0], probe.shape[1], probe.shape[2])
        print(f"[info] TIFF dir: N={len(tiffs)} shape={shape} dims={spatial_dims}")

        # sanity check: uniform shape
        for p in tiffs[1:]:
            arr = tif.imread(str(p))
            if arr.ndim != probe.ndim or arr.shape != probe.shape:
                raise SystemExit(f"[ERR] Shape mismatch: {p.name} has {arr.shape}, expected {probe.shape}")

        # cfg → sizes / ranges
        roi_size = parse_roi_size(roi_cfg, spatial_dims)

        # sampling range (applies equally to train and test; no spatial split now)
        if sample_range_cfg is None:
            sample_range = default_sample_range(shape)
        else:
            if len(sample_range_cfg) != spatial_dims:
                raise SystemExit(f"[ERR] sample_range dims != {spatial_dims}")
            sample_range = [list(map(int, r)) for r in sample_range_cfg]

        # ---- NEW: file-index-based split ----
        N = len(tiffs)
        n_train_files = int(math.floor(N * train_ratio))
        train_files = tiffs[:n_train_files]
        test_files  = tiffs[n_train_files:]

        print(f"[split-files] train_files={len(train_files)} (0..{n_train_files-1}), "
              f"test_files={len(test_files)} ({n_train_files}..{N-1})")

        # distribute counts across the chosen files
        tr_counts = distribute_counts(train_n, max(1, len(train_files)))
        te_counts = distribute_counts(test_n,  max(1, len(test_files)))

        # helpers to sample an ROI from a given array using the common sample_range
        if spatial_dims == 2:
            def make_get(arr):
                return lambda: sample_roi_2d(arr, sample_range, roi_size, filt)
        else:
            def make_get(arr):
                return lambda: sample_roi_3d(arr, sample_range, roi_size, filt)

        total_tr = total_te = 0

        # write train set from the first block of files
        for idx, fp in enumerate(train_files):
            arr = tif.imread(str(fp))
            get_roi = make_get(arr)
            stem = fp.stem
            total_tr += write_set(get_roi,
                                  tr_counts[idx],
                                  train_dir,
                                  lambda i, s=stem: f"{s}_tr_{i:04d}.tif",
                                  progress_every=100)

        # write test set from the remaining block of files
        for idx, fp in enumerate(test_files):
            arr = tif.imread(str(fp))
            get_roi = make_get(arr)
            stem = fp.stem
            total_te += write_set(get_roi,
                                  te_counts[idx],
                                  test_dir,
                                  lambda i, s=stem: f"{s}_te_{i:04d}.tif",
                                  progress_every=100)

        print(f"[OK] crop done (TIFF dir, file-index split) → train:{total_tr}  test:{total_te}")
        return 

    # Single TIFF
    vol = tif.imread(str(img_path))
    if vol.ndim not in (2, 3):
        raise SystemExit(f"[ERR] Unsupported TIFF ndim {vol.ndim} (expect 2 or 3)")

    spatial_dims = vol.ndim
    shape = vol.shape if spatial_dims == 2 else (vol.shape[0], vol.shape[1], vol.shape[2])
    print(f"[info] Single TIFF: shape={shape} dims={spatial_dims}")

    roi_size = parse_roi_size(roi_cfg, spatial_dims)
    split_axis = clamp_axis(split_axis_cfg, spatial_dims)

    if sample_range_cfg is None:
        sample_range = default_sample_range(shape)
    else:
        if len(sample_range_cfg) != spatial_dims:
            raise SystemExit(f"[ERR] sample_range dims != {spatial_dims}")
        sample_range = [list(map(int, r)) for r in sample_range_cfg]

    train_sr, test_sr = split_ranges(sample_range, split_axis, train_ratio)

    if spatial_dims == 2:
        get_tr = lambda: sample_roi_2d(vol, train_sr, roi_size, filt)
        get_te = lambda: sample_roi_2d(vol, test_sr,  roi_size, filt)
    else:
        get_tr = lambda: sample_roi_3d(vol, train_sr, roi_size, filt)
        get_te = lambda: sample_roi_3d(vol, test_sr,  roi_size, filt)

    tr_saved = write_set(get_tr, train_n, train_dir, lambda i: f"{i:05d}.tif")
    te_saved = write_set(get_te,  test_n,  test_dir,  lambda i: f"{i:05d}.tif", progress_every=100)
    print(f"[OK] crop done (single TIFF) → train:{tr_saved}  test:{te_saved}")


if __name__ == "__main__":
    main()