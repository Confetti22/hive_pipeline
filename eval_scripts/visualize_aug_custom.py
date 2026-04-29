import sys
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile as tiff

# Add project root to sys.path to allow imports from lib
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from lib.utils.preprocess_img import preprocess_uint16_for_imagenet
from lib.utils.augmentations import GaussianBlurSharpGPU, CentralCropGPU

def apply_gamma_contrast_brightness(x, alpha, beta, z):
    """
    Applies Unbiased Gamma, Contrast (alpha), and Brightness (beta).
    Logic follows UnbiasedGammaGPU in lib/utils/augmentations.py
    
    Note: preprocess_uint16_for_imagenet performs ImageNet normalization by default.
    Applying gamma to normalized data (which has negative values) requires clamping.
    """
    if z is not None and z != 0:
        inv_sqrt_2 = 1.0 / math.sqrt(2.0)
        gamma = math.log(0.5 + inv_sqrt_2 * z) / math.log(0.5 - inv_sqrt_2 * z)
    else:
        gamma = 1.0
    
    # x is expected to be (B, C, D, H, W)
    # Original code in lib/utils/augmentations.py clamps to 1e-6 to avoid NaN with power on negative values
    x_clamped = torch.clamp(x, min=1e-6)
    x_aug = alpha * torch.pow(x_clamped, gamma) + (beta if beta is not None else 0.0)
    return x_aug

def apply_blur_sharp(x, blur_sigma, sharp_sigma):
    """
    Applies Gaussian Blur or Sharpening.
    Logic follows GaussianBlurSharpGPU in lib/utils/augmentations.py
    """
    augmentor = GaussianBlurSharpGPU()
    
    # Handle D=1 case for 5D tensors to avoid padding issues in the library
    # b,c,d,h,w
    was_5d_d1 = False
    if x.ndim == 5 and x.shape[2] == 1:
        x = x.squeeze(2)
        was_5d_d1 = True
        
    if blur_sigma is not None:
        x = augmentor.apply_blur(x, blur_sigma)
    if sharp_sigma is not None:
        # Sharpening in lib/utils/augmentations.py: x = x + blending * (x - smoothed)
        # We use blending=1.0 for demonstration
        smoothed = augmentor.apply_blur(x, sharp_sigma)
        x = x + 1.0 * (x - smoothed)
        
    if was_5d_d1:
        x = x.unsqueeze(2)
        
    return x

def apply_affine(x, theta, d):
    """
    Applies Rotation (theta) and Translation (d).
    Logic follows RandomAffineGPU in lib/utils/augmentations.py
    """
    if theta is None and d is None:
        return x
    
    # Ensure 5D [B, C, D, H, W] for consistency, though we operate on H, W
    if x.ndim == 4:
        x = x.unsqueeze(2)
        was_4d = True
    else:
        was_4d = False
        
    B, C, D, H, W = x.shape
    device = x.device
    
    theta_val = theta if theta is not None else 0.0
    cos_t = math.cos(theta_val)
    sin_t = math.sin(theta_val)
    
    # Translation: d pixels to normalized coordinates [-1, 1]
    tx = 0.0
    ty = 0.0
    if d is not None:
        # In torch's F.affine_grid, tx/ty are normalized by half-width/half-height
        # Negative sign to move image in the direction of d
        tx = -2.0 * d / W
        ty = -2.0 * d / H
        
    matrix = torch.tensor([
        [cos_t, -sin_t, tx],
        [sin_t,  cos_t, ty]
    ], device=device).float().unsqueeze(0).repeat(B, 1, 1)
    
    # Since D=1, we treat it as 2D affine
    x_2d = x.squeeze(2) # [B, C, H, W]
    grid = F.affine_grid(matrix, x_2d.size(), align_corners=False)
    x_out = F.grid_sample(x_2d, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    if was_4d:
        return x_out
    else:
        return x_out.unsqueeze(2)

def main():

    images_uint16 = [
        tiff.imread('/home/confetti/data/t1779/z_slices_768_768/1123.tif'),
        tiff.imread('/home/confetti/data/t1779/z_slices_768_768/3924.tif'),
        tiff.imread('/home/confetti/data/rm009/v1_z_slices_768_768/1073.tif')
    ]
    cropper = CentralCropGPU(512)
    


    # Define the 5 augmentation combinations as requested
    # theta: rotation, d: translation, alpha: contrast, beta: brightness, Z: gamma, blur: sigma, sharp: sigma
    combinations = [
        {"theta": math.pi/4,   "d": 24,   "alpha": 0.9,  "beta": -0.05, "Z": None,  "blur": None, "sharp": None},
        {"theta": -math.pi/4,  "d": -24,  "alpha": 1.1,  "beta": 0.05,  "Z": -0.01, "blur": None, "sharp": None},
        {"theta": -math.pi/8,  "d": 12,   "alpha": 1.05, "beta": 0.10,  "Z": -0.02, "blur": None, "sharp": 0.25},
        {"theta": math.pi/8,   "d": 12,   "alpha": 0.95, "beta": -0.10, "Z": 0.01,  "blur": 0.25, "sharp": None},
        {"theta": None,        "d": None, "alpha": 1.02, "beta": None,  "Z": 0.05,  "blur": 0.8,  "sharp": None},
    ]

    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    row_labels = ['roi1', 'roi2', 'roi3']
    col_labels = ['original', 'A', 'B', 'C', 'D', 'E']
    
    for row_idx, img_np in enumerate(images_uint16):
        # 1. Preprocess (uint16 -> ImageNet normalized tensor [C, 1, H, W])
        tensor = preprocess_uint16_for_imagenet(img_np, make_3ch=False)
        tensor = tensor.unsqueeze(0) # [1, C, 1, H, W]
        
        # Calculate min/max from the original tensor for consistent normalization across the row
        v_min = tensor.min().item()
        v_max = tensor.max().item()
        v_range = v_max - v_min + 1e-8

        # Define viz helper with FIXED range based on original image
        def get_viz(t):
            arr = t[0, 0, 0].detach().cpu().numpy()
            # Normalize using the row's fixed min/max to preserve brightness/contrast changes
            return np.clip((arr - v_min) / v_range, 0, 1)

        # Original Column (Col 0)
        axes[row_idx, 0].imshow(get_viz(cropper(tensor)), cmap='gray')
        if row_idx == 0:
            axes[row_idx, 0].set_title(col_labels[0], fontsize=24)
        axes[row_idx, 0].set_ylabel(row_labels[row_idx], fontsize=24)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        # Hide spines but keep labels
        for spine in axes[row_idx, 0].spines.values():
            spine.set_visible(False)
        
        for col_idx, p in enumerate(combinations):
            # Apply augmentations in sequence
            x = tensor.clone()
            
            # 1. Gamma / Contrast / Brightness
            x = apply_gamma_contrast_brightness(x, p['alpha'], p['beta'], p['Z'])
            
            # 2. Blur / Sharp
            x = apply_blur_sharp(x, p['blur'], p['sharp'])
            
            # 3. Affine (Rotation / Translation)
            x = apply_affine(x, p['theta'], p['d'])

            # 4. Crop
            x = cropper(x)
            
            # Visualization Column (Col 1-5)
            axes[row_idx, col_idx + 1].imshow(get_viz(x), cmap='gray')
            if row_idx == 0:
                axes[row_idx, col_idx + 1].set_title(col_labels[col_idx + 1], fontsize=24)
            axes[row_idx, col_idx + 1].axis('off')

    plt.tight_layout()

    output_path = "data_aug_visualization.png"
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    main()
