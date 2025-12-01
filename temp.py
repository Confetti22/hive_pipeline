#%%
import numpy as np
import napari 
import tifffile as tif
from scipy.ndimage import zoom
parent_dir = '/home/confetti/data/dk/MD594/MD594'
idx_list = [317]
viewer = napari.Viewer()
# idx_list = [90,107,112,121] 90 is of 122 M
#TODO, read all the tif files that is bigger than 140MB and smaller than 170 BM

files = [f"{parent_dir}/{str(idx).zfill(3)}.tif" for idx in idx_list]
images = [tif.imread(fp) for fp in files]
for img, fp in zip(images, files):
    limits = tuple(np.percentile(img, (0, 99)))
    # viewer.add_image(img, name=fp, contrast_limits=limits)



roi_size =[3072,3072]
hp_offset=[8492,19969]
hp_img = tif.imread(f"{parent_dir}/317.tif")
hp_roi = hp_img[hp_offset[0]:hp_offset[0]+roi_size[0], hp_offset[1]:hp_offset[1]+roi_size[1]]
zoom_factors = (1536 / hp_roi.shape[0], 1536 / hp_roi.shape[1],1)
hp_roi = zoom(hp_roi, zoom=zoom_factors, order=1, prefilter=False)

# viewer.add_image(vii_roi, name='vii_roi_1', contrast_limits=(0, np.percentile(vii_roi, 99)))
# viewer.add_image(vii_roi2, name='vii_roi_2', contrast_limits=(0, np.percentile(vii_roi2, 99)))
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2, figsize=(20,10))
ax[0].imshow(hp_roi)
plt.show()
#%%
save_dir = '/home/confetti/data/t1779/scenes'
tif.imwrite(f'{save_dir}/dk_hp_roi.tif', hp_roi)

# %%
