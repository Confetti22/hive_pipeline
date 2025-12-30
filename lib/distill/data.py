from __future__ import annotations
from typing import List, Tuple, Optional

import tifffile as tiff
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from lib.utils.preprocess_img import preprocess_uint16_for_imagenet


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])  # Let autocast determine dtype
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])  # Let autocast determine dtype


class GrayTiffDataset(Dataset):
    def __init__(self, paths: List[str], crop_size: Optional[int] = None):
        """
        accept 2d tiff gray images
        
        Args:
            paths: List of paths to TIFF files
            crop_size: Optional central crop size. If None, no cropping is applied.
                      If specified, images will be centrally cropped to crop_size x crop_size
        """
        self.paths = list(paths)
        self.crop_size = crop_size

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
        
        # Apply central crop if specified
        if self.crop_size is not None:
            img = self._central_crop(img, self.crop_size)
            if idx == 0:
                print(f"After central crop ({self.crop_size}x{self.crop_size}): {img.shape}")
        
        # Reuse preprocess method ensures the input image is 2d
        x = preprocess_uint16_for_imagenet(img) # [C,D,H,W] if img is 2d, then D==1 , C==3
        x = x.squeeze(1) # [C,H,W]
        
        image_id = path
        return x, image_id # [C,H,W]
    
    def _central_crop(self, img: torch.Tensor, crop_size: int) -> torch.Tensor:
        """
        Apply central crop to the image.
        
        Args:
            img: Input image tensor
            crop_size: Size of the central crop (crop_size x crop_size)
            
        Returns:
            Centrally cropped image tensor
        """
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:
            h, w = img.shape
        
        # Calculate crop coordinates
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        end_h = start_h + crop_size
        end_w = start_w + crop_size
        
        # Ensure we don't go out of bounds
        start_h = max(0, start_h)
        start_w = max(0, start_w)
        end_h = min(h, end_h)
        end_w = min(w, end_w)
        
        # Apply crop
        if isinstance(img, torch.Tensor):
            cropped = img[start_h:end_h, start_w:end_w]
        else:
            cropped = img[start_h:end_h, start_w:end_w]
            cropped = torch.from_numpy(cropped)
        
        # If the crop is smaller than requested, pad with zeros
        if cropped.shape[-2] < crop_size or cropped.shape[-1] < crop_size:
            if isinstance(cropped, torch.Tensor):
                padded = torch.zeros((crop_size, crop_size), dtype=cropped.dtype)
            else:
                padded = torch.zeros((crop_size, crop_size), dtype=torch.float32)
            
            # Place the cropped image in the center of the padded image
            pad_h = (crop_size - cropped.shape[-2]) // 2
            pad_w = (crop_size - cropped.shape[-1]) // 2
            padded[pad_h:pad_h+cropped.shape[-2], pad_w:pad_w+cropped.shape[-1]] = cropped
            cropped = padded
        
        return cropped


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


