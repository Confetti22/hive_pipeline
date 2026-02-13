import sys
sys.path.append("/home/confetti/e5_workspace/hive1")

import os
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import tifffile as tif
import torch
from torch import Tensor
from torch.utils.data import Dataset

from lib.utils.preprocess_img import preprocess_uint16_for_imagenet


def _four_digit_key(p: Union[str, Path]) -> Union[int, str]:
    """Sort by first 4 digits in basename; fallback to whole name."""
    name = os.path.basename(str(p))
    head = name[:4]
    return int(head) if head.isdigit() else name


def _to_cdhw(arr: np.ndarray, make_3ch: bool = True) -> np.ndarray:
    """Convert array to (C,D,H,W)."""
    a = np.asarray(arr)
    if a.ndim == 2:                   # (H,W) -> (C=1 or 3, D=1, H, W)
        if make_3ch:
            a = np.stack([a, a, a], axis=0)   # (3,H,W)
        else:
            a = a[None, ...]                  # (1,H,W)
        a = a[:, None, ...]                   # (C,1,H,W)
        return a.astype(np.float32)

    if a.ndim == 3:
        if a.shape[-1] == 3:           # (H,W,3) -> (3,1,H,W)
            a = np.transpose(a, (2, 0, 1))
            a = a[:, None, ...]
            return a.astype(np.float32)
        else:                          # (D,H,W) -> (C=1 or 3, D,H,W)
            if make_3ch:
                a = np.stack([a, a, a], axis=0)
            else:
                a = a[None, ...]
            return a.astype(np.float32)

    if a.ndim == 4:                    # assume already (C,D,H,W)
        return a.astype(np.float32)

    raise ValueError(f"Unsupported image shape {a.shape}.")


class SegDataset(Dataset):
    def __init__(
        self,
        data_path_dir: str,
        mask_path_dir: str,
        use_ratio: float = 1.0,
        normalize: bool = True,
        valid_data: bool = False,
        valid_suffix: str = "_valid",
        make_3ch: bool = True,
        robust_percentiles: Tuple[float, float] = (1.0, 99.9),
        log_transform: bool = False,
        gamma: Optional[float] = None,
        shift_labels_to_zero: bool = True,
    ):
        """
        Paired ROI+Mask dataset with optional 'valid' dir switching.

        Args:
            data_path_dir: Directory of ROI images (.tif/.tiff).
            mask_path_dir: Directory of mask images (.tif/.tiff).
            use_ratio: Keep only the first `use_ratio * N` matched pairs.
            normalize: If True, preprocess ROI with ImageNet-style pipeline.
            valid_data: If True, read from dirs with `valid_suffix` appended.
                        E.g., '/path/rois' -> '/path/rois_valid'.
            valid_suffix: Suffix to append when `valid_data=True`.
            make_3ch: If True, replicate single-channel ROIs to 3 channels.
            robust_percentiles, log_transform, gamma: Passed to preprocess.
            shift_labels_to_zero: If True, mask = mask - 1 (for class 1..N -> 0..N-1).

        Returns:
            __getitem__ -> (roi, mask)
                roi:  torch.float32, (C,D,H,W)
                mask: torch.int64,   (D,H,W)
        """
        # Resolve (optionally) to *_valid dirs
        roi_dir = Path(data_path_dir + valid_suffix) if valid_data else Path(data_path_dir)
        msk_dir = Path(mask_path_dir + valid_suffix) if valid_data else Path(mask_path_dir)
        if not roi_dir.is_dir():
            raise FileNotFoundError(f"ROI dir not found: {roi_dir}")
        if not msk_dir.is_dir():
            raise FileNotFoundError(f"Mask dir not found: {msk_dir}")

        self.normalize = normalize
        self.make_3ch = make_3ch
        self.robust_percentiles = robust_percentiles
        self.log_transform = log_transform
        self.gamma = gamma
        self.shift_labels_to_zero = shift_labels_to_zero

        # Collect files and match by 4-digit prefix (or full name fallback)
        roi_files = sorted([p for p in roi_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")], key=_four_digit_key)
        msk_files = sorted([p for p in msk_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")], key=_four_digit_key)

        def _index(files):
            out = {}
            for p in files:
                k = _four_digit_key(p)
                if k not in out:
                    out[k] = p
            return out

        roi_idx = _index(roi_files)
        msk_idx = _index(msk_files)
        keys = sorted(set(roi_idx.keys()) & set(msk_idx.keys()), key=lambda k: (isinstance(k, str), k))
        if not keys:
            raise RuntimeError(f"No matching ROI/Mask pairs between:\n  {roi_dir}\n  {msk_dir}")

        keep_n = max(1, int(len(keys) * float(use_ratio)))
        keys = keys[:keep_n]
        self.pairs: List[Tuple[Path, Path]] = [(roi_idx[k], msk_idx[k]) for k in keys]

        print(
            f"###### init FixedDataset (valid={valid_data}) with amount={use_ratio}, "
            f"pairs={len(self.pairs)} #####\n"
            f"ROI dir = {roi_dir}\nMSK dir = {msk_dir}"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def _read_roi(self, p: Path) -> np.ndarray:
        raw = tif.imread(str(p))
        if self.normalize:
            proc = preprocess_uint16_for_imagenet(
                raw,
                make_3ch=self.make_3ch,
                robust_percentiles=self.robust_percentiles,
                log_transform=self.log_transform,
                gamma=self.gamma,
            )
        else:
            proc = raw.astype(np.float32)
        return _to_cdhw(proc, make_3ch=self.make_3ch)  # (C,D,H,W)

    def _read_mask(self, p: Path) -> np.ndarray:
        m = tif.imread(str(p))
        m = np.asarray(m)
        if m.ndim == 2:
            m = m[None, ...]                 # (1,H,W)

        if self.shift_labels_to_zero:
            m = m.astype(np.int64) - 1
        else:
            m = m.astype(np.int64)
        m = torch.from_numpy(m).long()  # (D,H,W)
        return m

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        roi_p, msk_p = self.pairs[idx]
        roi =  self._read_roi(roi_p)     # (C,D,H,W)
        mask = self._read_mask(msk_p)   # (D,H,W)
        return roi, mask


# ---- factory helpers (mirroring your old get_dataset name) ---- #

def get_dataset(
    data_path_dir: str,
    mask_path_dir: str,
    use_ratio: float = 1.0,
    **kwargs,
) -> SegDataset:
    """Training split (valid_data=False)."""
    return SegDataset(
        data_path_dir=data_path_dir,
        mask_path_dir=mask_path_dir,
        use_ratio=use_ratio,
        valid_data=False,
        **kwargs,
    )


def get_valid_dataset(
    data_path_dir: str,
    mask_path_dir: str,
    use_ratio: float = 1.0,
    **kwargs,
) -> SegDataset:
    """Validation split (valid_data=True)."""
    return SegDataset(
        data_path_dir=data_path_dir,
        mask_path_dir=mask_path_dir,
        use_ratio=use_ratio,
        valid_data=True,
        **kwargs,
    )