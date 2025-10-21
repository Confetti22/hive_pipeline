from __future__ import annotations
from typing import List, Tuple

import tifffile as tiff
import torch
from torch.utils.data import Dataset


class GrayTiffDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        img = tiff.imread(path)
        x = torch.from_numpy(img.astype("float32")) / 65535.0
        x = x.clamp(0, 1).unsqueeze(0)
        image_id = path
        return x, image_id


def to_rgb_for_vit(x_gray: torch.Tensor) -> torch.Tensor:
    return x_gray.repeat(1, 3, 1, 1)


