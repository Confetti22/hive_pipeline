import numpy as np
from scipy.ndimage import affine_transform,gaussian_filter
# from skimage import exposure
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import torch
from typing import Union, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Sequence


def generate_3d_volume_with_line(volume_shape, thickness, line_value, background_value):
    """
    Generate a 3D volume with a line stretching from the top-left to the bottom-right.

    Parameters:
    -----------
    volume_shape : tuple of int
        Shape of the 3D volume (depth, height, width).
    thickness : int
        Thickness of the line in pixels.
    line_value : int or float
        Value assigned to the pixels of the line.
    background_value : int or float
        Value assigned to the background pixels.

    Returns:
    --------
    np.ndarray
        3D volume with the specified line.
    """
    volume = np.full(volume_shape, background_value, dtype=np.float32)
    depth, height, width = volume_shape

    for z in range(depth):
        # Calculate the center coordinates of the line at each depth
        y = int(height * z / depth)
        x = int(width * z / depth)
        
        # Define the line thickness (in a square cross-section)
        for dy in range(-thickness // 2, thickness // 2 + 1):
            for dx in range(-thickness // 2, thickness // 2 + 1):
                yy = np.clip(y + dy, 0, height - 1)
                xx = np.clip(x + dx, 0, width - 1)
                volume[z, yy, xx] = line_value

    return volume

def random_rotation_3d(volume,lower_limit = -np.pi/6,upper_limit = np.pi/6):
    angles = generate_random_angles(lower_limit,upper_limit)
    rotation_matrix = generate_rotation_matrix(angles,order ='zyx')
    rotated_volume = rotated_volume(volume,rotation_matrix)
    return rotated_volume


def center_crop_3d(volume, crop_shape):
    """
    Perform a center crop on a 3D volume.

    Parameters:
    -----------
    volume : np.ndarray
        The 3D volume to be cropped, with shape (depth, height, width).
    crop_shape : tuple of int
        The desired crop shape (crop_depth, crop_height, crop_width).

    Returns:
    --------
    np.ndarray
        The cropped 3D volume.
    """
    if len(volume.shape) != 3:
        raise ValueError("Input volume must be a 3D array.")

    if any(cs > vs for cs, vs in zip(crop_shape, volume.shape)):
        raise ValueError("Crop shape must not exceed the dimensions of the input volume.")

    depth, height, width = volume.shape
    crop_depth, crop_height, crop_width = crop_shape

    # Calculate the start and end indices for each dimension
    start_d = (depth - crop_depth) // 2
    start_h = (height - crop_height) // 2
    start_w = (width - crop_width) // 2

    end_d = start_d + crop_depth
    end_h = start_h + crop_height
    end_w = start_w + crop_width

    # Perform the crop
    cropped_volume = volume[start_d:end_d, start_h:end_h, start_w:end_w]
    return cropped_volume

def random_gaussian_blur_3d(volume,P=0.8,sigma_range=(0.1, 2)):
    """
    Apply a random Gaussian blur to a 3D volume.
    
    Parameters:
    - volume (numpy.ndarray): Input 3D volume to be blurred.
    - sigma_range (tuple): Range of sigma values for the Gaussian blur.
    
    Returns:
    - blurred_volume (numpy.ndarray): The blurred 3D volume.
    """

    p = random.random()
    if p < P:
        # Generate random sigma within the specified range
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        print(f"sigma of blur is {sigma}")
        
        # Apply Gaussian filter
        blurred_volume = gaussian_filter(volume, sigma=sigma)
        return blurred_volume

    else :
        return volume



class RandomRotation3D:
    def __init__(self, lower_limit=-np.pi/6, upper_limit=np.pi/6,probability=0.8,v=False):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.probability = probability
        self.v = v

    def generate_random_angles(self,lower_limit, upper_limit):
        """
        Generate three random rotation angles within the specified range.

        Parameters:
        -----------
        lower_limit : float
            The lower limit of the angle range (in radians).
        upper_limit : float
            The upper limit of the angle range (in radians).

        Returns:
        --------
        tuple of floats
            Three random rotation angles (angle_x, angle_y, angle_z) in radians.
        
        Example:
        --------
        >>> lower_limit = -np.pi / 4  # -45 degrees in radians
        >>> upper_limit = np.pi / 4   # 45 degrees in radians
        >>> random_angles = generate_random_angles(lower_limit, upper_limit)
        >>> print("Random Rotation Angles (in radians):", random_angles)
        """
        if lower_limit > upper_limit:
            raise ValueError("Lower limit must be less than or equal to the upper limit.")
        
        # Generate three random angles uniformly distributed in the specified range
        angles = np.random.uniform(lower_limit, upper_limit, size=3)
        return tuple(angles)

    def generate_rotation_matrix(self,angles, order='xyz'):
        """
        Generate a 3D rotation transformation matrix by combining rotations 
        around the x, y, and z axes.

        Parameters:
        -----------
        angles : tuple of floats
            Rotation angles (in radians) around the x, y, and z axes, in the order specified.
            Example: (angle_x, angle_y, angle_z)
        
        order : str
            Order of rotations, specified as a string of 'x', 'y', and 'z'. Default is 'xyz'.
            Example: 'xyz', 'zyx', 'yxz', etc.

        Returns:
        --------
        numpy.ndarray
            A 3x3 rotation matrix representing the combined rotation.
        """
        if len(angles) != 3 or len(order) != 3:
            raise ValueError("Both 'angles' and 'order' must have exactly three elements.")
        
        # Rotation matrices for basic axes
        def Rx(theta):
            return np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
        
        def Ry(theta):
            return np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        
        def Rz(theta):
            return np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        
        # Map axis names to rotation functions
        axis_map = {'x': Rx, 'y': Ry, 'z': Rz}
        
        # Start with an identity matrix
        rotation_matrix = np.eye(3)
        
        # Apply rotations in the specified order
        for axis, angle in zip(order, angles):
            rotation_matrix = rotation_matrix @ axis_map[axis](angle)
        
        return rotation_matrix

    def rotate_volume(self, volume, rotation_matrix):
        """
        Apply a 3D rotation to a volume using a given rotation matrix.

        Parameters:
        -----------
        volume : numpy.ndarray
            3D image volume to be rotated (e.g., shape [128, 128, 128]).
        
        rotation_matrix : numpy.ndarray
            A 3x3 rotation matrix to apply.

        Returns:
        --------
        numpy.ndarray
            The rotated volume.
        """
        # Compute the center of the volume
        center = np.array(volume.shape) / 2

        # Define the full affine transformation matrix (4x4)
        # Add translation to rotate around the center of the volume
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = rotation_matrix  # Insert the rotation part
        translation = center - rotation_matrix @ center
        affine_matrix[:3, 3] = translation  # Add translation to align the rotation around the center

        # Apply the affine transformation
        rotated_volume = affine_transform(
            volume,
            matrix=rotation_matrix,
            offset=translation,
            order=3,  # Cubic interpolation
            mode='constant',  # Fill with 0s outside the volume
            cval=0.0
        )
        return rotated_volume

        
    def __call__(self, volume):
        if random.random() < self.probability:
            angles = self.generate_random_angles(self.lower_limit, self.upper_limit)
            rotation_matrix = self.generate_rotation_matrix(angles, order='zyx')
            rotated_volume = self.rotate_volume(volume, rotation_matrix)
            return rotated_volume
        if self.v:
            print(f"this time did not rotated")
        return volume


class CenterCrop3D:
    def __init__(self, crop_shape):
        self.crop_shape = crop_shape

    def __call__(self, volume):
        if len(volume.shape) != 3:
            raise ValueError("Input volume must be a 3D array.")

        if any(cs > vs for cs, vs in zip(self.crop_shape, volume.shape)):
            raise ValueError("Crop shape must not exceed the dimensions of the input volume.")

        depth, height, width = volume.shape
        crop_depth, crop_height, crop_width = self.crop_shape

        start_d = (depth - crop_depth) // 2
        start_h = (height - crop_height) // 2
        start_w = (width - crop_width) // 2

        end_d = start_d + crop_depth
        end_h = start_h + crop_height
        end_w = start_w + crop_width

        return volume[start_d:end_d, start_h:end_h, start_w:end_w]




class RandomGaussianBlur3D:
    def __init__(self, sigma_range=(0.1, 2), probability=0.8,v=False):
        self.sigma_range = sigma_range
        self.probability = probability
        self.v = v

    def __call__(self, volume):
        if random.random() < self.probability:
            sigma = np.random.uniform(*self.sigma_range)
            if self.v:
                print(f"sigma of blur is {sigma}")
            return gaussian_filter(volume, sigma=sigma)
        if self.v:
            print(f"this time did not blur")
        return volume



class UnbiasedGammaAugmentation:
    def __init__(self, alpha_range=(0.9,1.1), beta_range=(-0.1,0.1), z_range =(-0.05, 0.05), v=False):
        self.alpha_range = alpha_range
        self.beta_range = beta_range 
        self.z_range = z_range
        self.unbiased_gamma = lambda z: np.log(0.5 + 2 ** (-0.5) * z) / np.log(0.5 - 2 ** (-0.5) * z)
        self.v = v

    def __call__(self, volume):

        z = np.random.uniform(*self.z_range)
        alpha = np.random.uniform(*self.alpha_range)
        beta = np.random.uniform(*self.beta_range)
        gamma = self.unbiased_gamma(z)
        volume = alpha * volume ** gamma + beta

        return volume



class Gaussin_blur_sharp_Augmentation:
    def __init__(self, sigma_range =(0.125, 1), blending_range=(0.5, 1.5), prob=(0.25,0.25), v=False):
        self.sigma_range = sigma_range,
        self.blending_range = blending_range
        self.prob = prob
        self.v = v


    def __call__(self, volume):
        if random.random() < self.prob[0]:
            sigma = np.random.uniform(*self.sigma_range)
            volume = gaussian_filter(volume, sigma=sigma)

        if random.random() < self.prob[1]:
            sigma = np.random.uniform(*self.sigma_range)
            blending = np.random.uniform(*self.blending_range)
            smoothed = gaussian_filter(volume, sigma=sigma)
            volume = volume + blending * (smoothed - volume)
       
        return volume


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Sequence

class UnbiasedGammaGPU(nn.Module):
    def __init__(self, alpha_range=(0.9, 1.1), beta_range=(-0.1, 0.1), z_range=(-0.05, 0.05), v=False):
        super().__init__()
        self.alpha_range = alpha_range
        self.beta_range = beta_range 
        self.z_range = z_range
        self.v = v
        self.INV_SQRT_2 = 1.0 / math.sqrt(2.0)

    def _get_gamma(self, z):
        # z is tensor of shape (B, 1, 1, ...)
        numerator = torch.log(0.5 + self.INV_SQRT_2 * z)
        denominator = torch.log(0.5 - self.INV_SQRT_2 * z)
        return numerator / denominator

    def forward(self, x):
        """
        x: Tensor (B, C, H, W) or (B, C, D, H, W)
        """
        B = x.shape[0]
        device = x.device
        ndim = x.ndim # 4 for 2D, 5 for 3D

        # 1. Sample parameters for the ENTIRE batch at once
        # Shape: (B,)
        z = torch.empty(B, device=device).uniform_(*self.z_range)
        alpha = torch.empty(B, device=device).uniform_(*self.alpha_range)
        beta = torch.empty(B, device=device).uniform_(*self.beta_range)

        # 2. Reshape for Broadcasting: (B, 1, 1, 1) or (B, 1, 1, 1, 1)
        # We need to append (ndim - 1) ones to match (C, [D], H, W)
        view_shape = [B] + [1] * (ndim - 1)
        
        z = z.view(*view_shape)
        alpha = alpha.view(*view_shape)
        beta = beta.view(*view_shape)

        # 3. Compute Gamma
        gamma = self._get_gamma(z)

        # 4. Apply transformation
        # Clamp to avoid NaN
        x_clamped = torch.clamp(x, min=1e-6)
        x_aug = alpha * torch.pow(x_clamped, gamma) + beta
        
        if self.v:
            print(f"Batch Gamma: Applied {B} different transforms.")
            
        return x_aug

class GaussianBlurSharpGPU(nn.Module):
    def __init__(self, sigma_range=(0.125, 1.0), blending_range=(0.5, 1.5), prob=(0.3, 0.3), v=False):
        super().__init__()
        self.sigma_range = sigma_range
        self.blending_range = blending_range
        self.prob = prob
        self.v = v

    def _get_gaussian_kernel(self, sigma, dim, device):
        # Kernel generation logic remains the same (generating one kernel)
        k_size = int(math.ceil(4.0 * sigma)) | 1 
        center = k_size // 2
        coords = torch.arange(k_size, dtype=torch.float32, device=device) - center
        kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        
        if dim == 2:
            kernel = kernel_1d[:, None] * kernel_1d[None, :]
        elif dim == 3:
            kernel = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
        
        return kernel / kernel.sum()

    def apply_blur(self, x, sigma):
        # x is (B, C, H, W) or (B, C, D, H, W)
        dim = x.ndim - 2 # Subtract B and C -> 2 or 3
        device = x.device
        C = x.shape[1] 
        
        kernel = self._get_gaussian_kernel(sigma, dim, device)
        
        # Prepare kernel for Group Conv
        # Groups = C (apply same kernel to each channel independently)
        # Weights: (C, 1, K, K...)
        kernel = kernel.expand(C, 1, *kernel.shape)
        
        pad_size = kernel.shape[-1] // 2
        
        if dim == 2:
            x_padded = F.pad(x, (pad_size,)*4, mode='reflect')
            # groups=C allows handling inputs of shape (B, C, H, W) correctly
            # It treats the batch dimension B naturally
            x_blur = F.conv2d(x_padded, kernel, groups=C)
        elif dim == 3:
            x_padded = F.pad(x, (pad_size,)*6, mode='reflect')
            x_blur = F.conv3d(x_padded, kernel, groups=C)
            
        return x_blur

    def forward(self, x):
        # Optimization: We apply the SAME sigma to the whole batch for speed.
        # Doing per-sample sigma in a batch requires very complex grouped convolutions 
        # (groups = B*C) or a slow loop. 
        
        was_5d_d1 = False
        if x.ndim == 5 and x.shape[2] == 1:
            x = x.squeeze(2)
            was_5d_d1 = True

        # Case 1: Blur
        if torch.rand(1) < self.prob[0]:
            sigma = getattr(self, "sigma_fixed", None) or torch.empty(1).uniform_(*self.sigma_range).item()
            x = self.apply_blur(x, sigma)
            if self.v:
                print(f"blur: sigma of blur is {sigma}")

        # Case 2: Sharpen/Blend
        if torch.rand(1) < self.prob[1]:
            sigma = torch.empty(1).uniform_(*self.sigma_range).item()
            blending = torch.empty(1).uniform_(*self.blending_range).item()
            
            smoothed = self.apply_blur(x, sigma)
            x = x + blending * (x - smoothed)
            if self.v:
                print(f"sharp: blending is {blending}")
                
        if was_5d_d1:
            x = x.unsqueeze(2)
            
        return x

class RandomAffineGPU(nn.Module):
    def __init__(self, angle_range_deg=30, translate_range_pix=24, mirror_prob=0.5, p=0.8, interpolation='bilinear', padding='zeros',v=False):
        super().__init__()
        self.angle_rad = math.radians(angle_range_deg)
        self.trans_pix = translate_range_pix
        self.mirror_prob = mirror_prob
        self.probability = p
        self.mode = interpolation
        self.padding = padding
        self.v = v

    def _get_2d_params(self, N, shape, device):
        # Generate N sets of parameters (one per batch item)
        H, W = shape
        
        # 1. Rotation: (N,)
        theta = (torch.rand(N, device=device) * 2 - 1) * self.angle_rad
        c, s = torch.cos(theta), torch.sin(theta)

        # 2. Translation: (N,)
        d = torch.rand(N, device=device) * self.trans_pix
        phi = torch.rand(N, device=device) * 2 * math.pi
        tx = -2.0 * (d * torch.cos(phi)) / W
        ty = -2.0 * (d * torch.sin(phi)) / H

        # 3. Mirror
        do_mirror = torch.rand(N, device=device) < self.mirror_prob
        sy = torch.where(do_mirror, torch.tensor(-1.0, device=device), torch.tensor(1.0, device=device))
        sx = torch.ones(N, device=device)

        # 4. Build Matrices (N, 2, 3)
        # Expand dims to match (N,) -> (N, 1) for broadcasting logic if needed, 
        # but torch.stack handles vector inputs fine
        row1 = torch.stack([sx * c, -sy * s, tx], dim=1)
        row2 = torch.stack([sx * s,  sy * c, ty], dim=1)
        return torch.stack([row1, row2], dim=1)

    def _get_3d_params(self, N, shape, device):
        D, H, W = shape
        
        # 1. Rotations (N,)
        rx = (torch.rand(N, device=device) * 2 - 1) * self.angle_rad
        ry = (torch.rand(N, device=device) * 2 - 1) * self.angle_rad
        rz = (torch.rand(N, device=device) * 2 - 1) * self.angle_rad
        
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)

        # 2. Translation
        d = torch.rand(N, device=device) * self.trans_pix
        v = torch.randn(N, 3, device=device)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-6) * d.unsqueeze(1)
        
        tz = -2.0 * v[:, 0] / D
        ty = -2.0 * v[:, 1] / H
        tx = -2.0 * v[:, 2] / W

        # 3. Mirror
        do_mirror = torch.rand(N, device=device) < self.mirror_prob
        scale_y = torch.where(do_mirror, torch.tensor(-1.0, device=device), torch.tensor(1.0, device=device))

        # 4. Build Matrix (N, 3, 4)
        r11 = cy * cz
        r12 = cz * sx * sy - cx * sz
        r13 = cx * cz * sy + sx * sz
        
        r21 = scale_y * (cy * sz)
        r22 = scale_y * (cx * cz + sx * sy * sz)
        r23 = scale_y * (-cz * sx + cx * sy * sz)
        
        r31 = -sy
        r32 = cy * sx
        r33 = cx * cy

        row1 = torch.stack([r11, r12, r13, tx], dim=1)
        row2 = torch.stack([r21, r22, r23, ty], dim=1)
        row3 = torch.stack([r31, r32, r33, tz], dim=1)
        
        return torch.stack([row1, row2, row3], dim=1)

    def forward(self, x):
        """x: (B, C, H, W) or (B, C, D, H, W)"""
        # Apply probability check to the WHOLE batch? 
        # Or apply identity matrix to some items?
        # For efficiency here, we apply if random < p, else return. 
        # (To do per-item probability requires masking the grid, which is possible but complex)
        if torch.rand(1) > self.probability:
            return x

        B = x.shape[0]
        device = x.device
        ndim = x.ndim - 2 # 2 or 3

        if ndim == 2:
            theta = self._get_2d_params(B, x.shape[2:], device)
        elif ndim == 3:
            theta = self._get_3d_params(B, x.shape[2:], device)
        else:
            raise ValueError("Input must be 4D or 5D")

        # theta is (B, 2, 3) or (B, 3, 4)
        # x is (B, C, ...)
        # affine_grid handles B automatically
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_out = F.grid_sample(x, grid, mode=self.mode, padding_mode=self.padding, align_corners=False)
        if self.v:
            print(f"affine transformation matrix_shape{theta.shape} is applied")
        
        return x_out

class CentralCropGPU(nn.Module):
    def __init__(self, size: Union[int, Sequence[int]], v=False):
        super().__init__()
        self.size = size
        self.v = v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) or (B, C, D, H, W)"""
        input_shape = x.shape
        # spatial dims are now starting from index 2
        spatial_dims = input_shape[2:] 
        ndim_spatial = len(spatial_dims)

        # Parse size
        if isinstance(self.size, int):
            crop_size = (self.size,) * ndim_spatial
        else:
            crop_size = self.size

        # Slices: Keep Batch(0) and Channel(1)
        slices = [slice(None), slice(None)] 
        
        for dim_len, crop_len in zip(spatial_dims, crop_size):
            if crop_len > dim_len:
                start = 0
                end = dim_len
            else:
                start = (dim_len - crop_len) // 2
                end = start + crop_len
            slices.append(slice(start, end))

        if self.v:
            print(f"central_crop with size {self.size} is applied")
        return x[tuple(slices)]



class GPUAugmentations(nn.Module):
    def __init__(self,size: Union[int, Sequence[int],None],v=False, affine =True):
        super().__init__()
        # In a real app, you would import the classes we wrote earlier here.
        # For this toy example, we use PyTorch's native random flip/rotate 
        # to simulate the "heavy lifting" we implemented previously.
        self.gamma = UnbiasedGammaGPU(v=v)
        self.blur = GaussianBlurSharpGPU(v=v)
    
        self.affine = RandomAffineGPU(v=v) if affine else None

        self.crop = CentralCropGPU(size,v=v) if size else None


    def forward(self, x):
        """
        Input: Batch of Tensors (N, C, D, H, W) on GPU
        Output: Augmented Batch on GPU
        """
        with torch.no_grad(): # Disable gradients for augmentation to save memory
            x = self.gamma(x)
            x = self.blur(x)
            if self.affine:
                x = self.affine(x)
            if self.crop != None:
                x = self.crop(x)
        return x