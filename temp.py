#%%
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
import napari

save_dir="/home/confetti/data/t1779/scenes"
os.makedirs(save_dir,exist_ok=True)

image_path = "/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Mouse_Brain/20210131_ZSS_USTC_THY1-YFP_1779_1/Reconstruction_1.0/z00000_c1.ims.part"

level = 0
channel = 2
roi_size = [64,1536,1536]
offsets = [7000-26,1840,4000]

ims_vol = Ims_Image(image_path, channel=channel)
D,H,W= ims_vol.rois[level][3:]
vol = ims_vol.from_roi((offsets+roi_size ),level = 0)
#%%

tif.imwrite("/home/confetti/data/t1779/scenes/visa2_1536_1536_64.tif",vol)
mip_roi = np.max(vol[12:-12],axis =0)
tif.imwrite("/home/confetti/data/t1779/scenes/mip_visa2_1536_1536_64.tif",mip_roi)
mask = np.zeros_like(mip_roi,dtype=np.uint8) 
print(f"{mip_roi.shape}")

#%%
viewer = napari.Viewer()
viewer.add_image(mip_roi)
viewer.add_labels(mask)
napari.run()


# %%
