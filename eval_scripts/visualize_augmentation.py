#%%
import sys
import os
import torch
import numpy as np

# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
import matplotlib.pyplot as plt
from pathlib import Path

from lib.utils.augmentations import RandomAffineGPU, GaussianBlurSharpGPU, UnbiasedGammaGPU,CenterCrop3D, CentralCropGPU
from lib.distill.data import GrayTiffDataset 


# --- Verification ---
# 1. Setup Data
data_root = Path('/home/confetti/data/t1779/z_slices_512_512')
if data_root.is_dir():
    train_paths = sorted([str(p) for p in data_root.glob("*.tif*")])
else:
    train_paths = []
    print(f"Warning: {data_root} not found or not a directory.")


if not train_paths:
    # Create dummy data if path doesn't exist
    print("Creating dummy data for verification...")
    dummy_img = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)
    import tifffile
    tifffile.imwrite("dummy_test.tif", dummy_img)
    train_paths = ["dummy_test.tif"]

ds = GrayTiffDataset(train_paths)
img, img_id = ds[0]  # [C, H, W]
print(f"Loaded image {img_id}, shape: {img.shape}, dtype: {img.dtype}")

# 2. Setup Device & Augmentations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}")

img_gpu = img.to(device)

#%%
import tifffile as tif
img = tif.imread('/home/confetti/data/t1779/scenes_o/3_1_roi.tif')
print(img.shape)
img = img/255.0
img_gpu = torch.from_numpy(img).permute(2,0,1).to(device)
img_gpu = img_gpu.to(torch.float32)
print(img_gpu.shape, img_gpu.max(),img_gpu.min())

#%%
# Initialize with verbose=True to print parameters
aug_affine = RandomAffineGPU()
aug_blur = GaussianBlurSharpGPU( prob=(0.5, 0.5),v=True) # Force prob for demo
aug_gamma = UnbiasedGammaGPU(v=True,beta_range=(0,0), alpha_range=(1,1),z_range=(-0.5,0.5))
cropper_2d = CentralCropGPU(size=1024, v=True)
# 3. Apply Transforms
print("\n--- RandomAffineGPU ---")
out_affine = aug_affine(img_gpu.clone())
cropped = cropper_2d(out_affine)


print("\n--- GaussianBlurSharpGPU ---")
out_blur = aug_blur(img_gpu.clone())

print("\n--- UnbiasedGammaGPU ---")
out_gamma = aug_gamma(img_gpu.clone())

# 4. Visualize
def to_numpy(t):
    return t.detach().cpu().numpy()[0] # Take 1st channel for viz

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(to_numpy(img_gpu), ); axes[0].set_title("Original")
axes[1].imshow(to_numpy(cropped), ); axes[1].set_title("Affine")
axes[2].imshow(to_numpy(out_blur), ); axes[2].set_title("Blur/Sharp")
axes[3].imshow(to_numpy(out_gamma), ); axes[3].set_title("Gamma")

for ax in axes: ax.axis('off')
plt.show()


# %%
