import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
from lib.utils.preprocess_img import preprocess_uint16_for_imagenet, preprocess_uint8rgb_for_imagenet

class SparseLabelSegDataset(Dataset):
    """Dataset from a single ROI and sparse integer labels.
    update: now always accept patch_size of (d,h,w) for both 2D and 3D inputs.

    """
    def __init__(self,
                 image: np.ndarray,
                 labels: np.ndarray,
                 dims: int = 2,
                 patch_size: Tuple[int, ...] | None = None,
                 imagenet_preproc: bool = False,
                 max_samples: int = 512):
        super().__init__()
        self.image = image
        self.labels = labels
        self.dims = dims
        self.patch_size = patch_size
        self.samples: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []  # (lo, hi) for slicing
        self.imagenet_preproc = imagenet_preproc

        # the location of all the labeled points
        coords = np.column_stack(np.nonzero(labels))  # N x (2|3)
        if coords.size == 0:
            return


        if dims == 2:
            H, W  = image.shape[:2]
            ph,pw = patch_size[1:]
            if (H == ph and W == pw)  or (H < ph)  or  (W < pw):
                # Use full image as one sample if reasonably sized
                self.samples.append(((0, 0), (H, W)))
            else:
                for y, x in coords[::max(1, len(coords)//max_samples)]:
                    y0 = max(0, y - ph//2); y1 = min(H, y0 + ph); y0 = y1 - ph
                    x0 = max(0, x - pw//2); x1 = min(W, x0 + pw); x0 = x1 - pw
                    if y0 < 0 or x0 < 0:
                        continue
                    self.samples.append(((y0, x0), (y1, x1)))
        else:
            D, H, W = image.shape[:3]
            pd, ph, pw = patch_size

            if (pd ==D and ph == H and pw == W) or (pd > D) or (ph > H) or (pw > W): 
                # Use full image as one sample if reasonably sized
                self.samples.append(((0, 0, 0), (D, H, W)))
            else:
                step = max(1, len(coords)//max_samples)
                for z, y, x in coords[::step]:
                    z0 = max(0, z - pd//2); z1 = min(D, z0 + pd); z0 = z1 - pd
                    y0 = max(0, y - ph//2); y1 = min(H, y0 + ph); y0 = y1 - ph
                    x0 = max(0, x - pw//2); x1 = min(W, x0 + pw); x0 = x1 - pw
                    if z0 < 0 or y0 < 0 or x0 < 0:
                        continue
                    self.samples.append(((z0, y0, x0), (z1, y1, x1)))
    


    def __len__(self) -> int:
        return int(len(self.samples))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lo, hi = self.samples[idx]
        if self.dims == 2:
            (y0, x0), (y1, x1) = lo, hi
            img = self.image[y0:y1, x0:x1]
            lab = self.labels[y0:y1, x0:x1]

            if self.imagenet_preproc:
                if len(img.shape) ==2: 
                    x = preprocess_uint16_for_imagenet(img)  # [C,H,W]
                else:
                    x = preprocess_uint8rgb_for_imagenet(img)
            else:
                x = torch.from_numpy(img.astype(np.float32))[None]  # [1,H,W]
            y = torch.from_numpy(lab.astype(np.int64))         # [H,W]
        else:
            (z0, y0, x0), (z1, y1, x1) = lo, hi
            img = self.image[z0:z1, y0:y1, x0:x1]
            if len(self.labels.shape) ==3:
                lab = self.labels[z0:z1, y0:y1, x0:x1]
            else:
                lab = self.labels[ y0:y1, x0:x1]

            if self.imagenet_preproc:
                if len(img.shape) ==3:
                    x = preprocess_uint16_for_imagenet(img) # [C,D,H,W]
                else:
                    x = preprocess_uint8rgb_for_imagenet(img)
            else:
                x = torch.from_numpy(img.astype(np.float32))[None]  # [1,D,H,W]
            y = torch.from_numpy(lab.astype(np.int64))          # [D,H,W]
        
        # remap label from 0 to N-1
        return x, y - 1


