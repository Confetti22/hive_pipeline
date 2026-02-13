from __future__ import annotations
from typing import List, Tuple, Optional

import tifffile as tiff
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from lib.utils.preprocess_img import preprocess_uint16_for_imagenet

from lib.utils.augmentations import RandomAffineGPU, GaussianBlurSharpGPU, UnbiasedGammaGPU, CentralCropGPU


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])  # Let autocast determine dtype
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])  # Let autocast determine dtype


class GrayTiffDataset(Dataset):
    def __init__(self, paths: List[str]):
        """
        accept 2d tiff gray images
        
        Args:
            paths: List of paths to TIFF files
        """
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Load and preprocess a grayscale TIFF image.
        
        Automatically rescales the image to [0,1] range based on the maximum value
        found in the image, rather than assuming a fixed uint16 range.
        Optionally applies central crop preprocessing.
        
        Args:
            idx: Index of the image to load
            
        Returns:
            Tuple of (preprocessed_tensor, image_path)
        """
        path = self.paths[idx]
        img = tiff.imread(path)

        if len(img.shape) == 3:  # only load the first slice if img is a 3d volume 
            img = img[0]
        
        
        # Reuse preprocess method ensures the input image is 2d
        x = preprocess_uint16_for_imagenet(img) # [C,D,H,W] if img is 2d, then D==1 , C==3
        x = x.squeeze(1) # [C,H,W]

        
        image_id = path
        return x, image_id # [C,H,W]
    


def to_rgb_for_vit(x_gray: torch.Tensor) -> torch.Tensor:
    """
    Convert grayscale tensor to RGB for Vision Transformer.
    
    Args:
        x_gray: Input tensor of shape [B, C, H, W] where C can be 1 (grayscale) or 3 (already RGB)
        
    Returns:
        RGB tensor of shape [B, 3, H, W]
    """
    if x_gray.shape[1] == 1:
        # Single channel grayscale -> repeat to 3 channels
        return x_gray.repeat(1, 3, 1, 1)
    elif x_gray.shape[1] == 3:
        # Already 3 channels (RGB) -> return as is
        return x_gray
    else:
        raise ValueError(f"Expected input with 1 or 3 channels, got {x_gray.shape[1]} channels")


