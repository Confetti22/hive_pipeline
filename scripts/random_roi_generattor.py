"""
Random ROI Generator for Autoencoder Training

This script generates random regions of interest (ROIs) from either:
1. A single .ims file (3D microscopy data)
2. A directory containing multiple 2D TIFF files (stack of 2D images)

The script extracts random 2D patches that pass an entropy filter and saves them
as individual TIFF files for autoencoder training.

Usage:
    python randomly_generate_cropped_roi4ae.py [image_path]
    
    Where image_path can be:
    - Path to a .ims file: "/path/to/data.ims"
    - Path to a directory containing TIFF files: "/path/to/tiff/directory"
    
    If no argument is provided, the script uses the hardcoded path below.
"""

import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from helper.image_reader import Ims_Image
from skimage.measure import shannon_entropy
import numpy as np
import tifffile as tif
import os
import glob
import random

class TiffDirectoryReader:
    """
    Class to handle reading and processing 2D TIFF files from a directory
    """
    def __init__(self, tiff_dir):
        self.tiff_dir = tiff_dir
        # Get all TIFF files in the directory
        self.tiff_files = sorted(glob.glob(os.path.join(tiff_dir, "*.tif*")))
        if not self.tiff_files:
            raise ValueError(f"No TIFF files found in directory: {tiff_dir}")
        
        # Load first image to get dimensions
        try:
            first_img = tif.imread(self.tiff_files[0])
            if len(first_img.shape) != 2:
                raise ValueError(f"Expected 2D TIFF files, but got {len(first_img.shape)}D image")
            self.H, self.W = first_img.shape
            self.D = len(self.tiff_files)
            
            print(f"Found {self.D} TIFF files with dimensions {self.H}x{self.W}")
        except Exception as e:
            raise ValueError(f"Error reading TIFF files from {tiff_dir}: {e}")
    
    def get_random_roi(self, filter_func, roi_size, level=0, skip_gap=False, sample_range=None, margin=0):
        """
        Get a random ROI from the TIFF stack
        """
        # For 2D TIFFs, we'll extract a 2D ROI from a random slice
        # roi_size should be (1, height, width) for 2D extraction
        if roi_size[0] != 1:
            print("Warning: For 2D TIFF files, roi_size[0] should be 1. Setting to 1.")
            roi_size = (1, roi_size[1], roi_size[2])
        
        foreground_sample_flag = False
        
        # Set up sampling ranges
        if sample_range is None:
            sample_lb = [0, 0, 0]
            sample_rb = [self.D, self.H, self.W]
        else:
            sample_lb = [idx_range[0] for idx_range in sample_range]
            sample_rb = [idx_range[1] for idx_range in sample_range]
        
        while not foreground_sample_flag:
            # Random slice selection
            z_idx = random.randint(sample_lb[0] + margin, sample_rb[0] - roi_size[0] - margin)
            y_idx = random.randint(sample_lb[1] + margin, sample_rb[1] - roi_size[1] - margin)
            x_idx = random.randint(sample_lb[2] + margin, sample_rb[2] - roi_size[2] - margin)
            
            # Load the selected slice
            img = tif.imread(self.tiff_files[z_idx])
            
            # Extract ROI
            roi = img[y_idx:y_idx+roi_size[1], x_idx:x_idx+roi_size[2]]
            
            # Reshape to match expected format (1, H, W)
            roi = roi.reshape(roi_size)
            roi = np.squeeze(roi)
            
            # Apply filter
            foreground_sample_flag = filter_func(roi)
        
        return roi, [z_idx, y_idx, x_idx]

def entropy_filter(l_thres=1.4, h_thres=100):
    def _filter(img):
        entropy=shannon_entropy(img)
        if (entropy>= l_thres) and (entropy <= h_thres):
            print(f"entrop of the roi is {entropy}")
            return True
        else:
            return False
    return _filter


# save_dir="/home/confetti/data/t1779/z_slices_512_512"
save_dir="/home/confetti/data/rm009/v1_z_slices_768_768"
os.makedirs(save_dir,exist_ok=True)


image_path = "/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Macaque_Brain/RM009_2/Analysis/ROIReconstruction/ROIImage/z13750_c1.ims"
# image_path = "/path/to/your/tiff/directory"

# You can also pass the image_path as a command line argument
if len(sys.argv) > 1:
    image_path = sys.argv[1]
    print(f"Using command line argument for image_path: {image_path}")

level = 0
channel = 0
roi_size =(1,768,768)
amount = 2**15

cnt = 1

# Determine input type and initialize appropriate reader
if os.path.isfile(image_path) and str(image_path).lower().endswith(('.ims','.ims.part')):
    # Handle .ims file
    print("Processing .ims file...")
    ims_vol = Ims_Image(image_path, channel=channel)
    D,H,W= ims_vol.rois[level][3:]
    margin = 10 

    # Generate random (x, y, z) locations within the given range
    d_near = 64
    #for whole brian
    # lz, hz = d_near + margin +int(D//4),  int(D*3/4) - d_near - margin
    # ly, hy = d_near + margin +int(H//4),  int(H*3/4) - d_near - margin
    # lx, hx = d_near + margin +int(W//4),  int(W//2) - d_near - margin
    #for part brain
    lz, hz = d_near + margin ,  int(D) - d_near - margin
    ly, hy = d_near + margin ,  int(H) - d_near - margin
    lx, hx = d_near + margin ,  int(W) - d_near - margin

    # sample_range = [[1000,17800],[100,15000],[100,16000]] #ae sample range for rm009
    sample_range = [[lz,hz], [ly,hy],[lx,hx]]
    
    vol_shape = ims_vol.info[level]['data_shape']
    reader = ims_vol
    
elif os.path.isdir(image_path):
    pass
#     # Handle TIFF directory
#     print("Processing TIFF directory...")
#     reader = TiffDirectoryReader(image_path)
#     D, H, W = reader.D, reader.H, reader.W
#     margin = 10

#     # Generate random (x, y, z) locations within the given range
#     d_near = 64
#     lz, hz = d_near + margin +int(D//4),  int(D*3/4) - d_near - margin
#     ly, hy = d_near + margin +int(H//6),  int(H*5/6) - d_near - margin
#     lx, hx = d_near + margin +int(W//6),  int(W*5/6) - d_near - margin

#     sample_range = [[lz,hz], [ly,hy],[lx,hx]]
    
else:
    raise ValueError(f"Input path must be either a .ims file or a directory containing TIFF files. Got: {image_path}")

print(f"Volume dimensions: D={D}, H={H}, W={W}")
print(f"Sample range: {sample_range}")

while cnt < amount:
    roi, indexs = reader.get_random_roi(
        filter=entropy_filter(l_thres=2.7),
        roi_size=roi_size,
        level=0,
        skip_gap=False,
        sample_range=sample_range,
        margin=0
    )
    roi = np.squeeze(roi)
    file_name = f"{save_dir}/{cnt:04d}.tif"
    tif.imwrite(file_name, roi)
    print(f"{file_name} has been saved ")
    cnt = cnt + 1

