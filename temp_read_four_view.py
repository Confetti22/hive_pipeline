#%%
import h5py
import dask.array as da
import napari
import numpy as np
from helper.image_reader import Ims_Image
import matplotlib.pyplot as plt

    
#data handler for four-view reconstruction data(deconv)
deconv_file_path = '../../e5_data/t1779/fourview/22.h5'

deconv_f = h5py.File(deconv_file_path, 'r')
deconv_dataset = deconv_f['data']
print(f"Loading {deconv_file_path}")
print(f"Dataset shape: {deconv_dataset.shape}")

# [0, Z, Y, X] downsampling by 2 in Z, 10 in Y, 10 in X
data_subset = deconv_dataset[0,::2,::10,::10]

zy_mip = np.max(data_subset, axis=2) # Max over X
zx_mip = np.max(data_subset, axis=1) # Max over Y
xy_mip = np.max(data_subset, axis=0) # Max over Z

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(zy_mip, cmap='gray')
plt.title('ZY MIP')
plt.xlabel('Y')
plt.ylabel('Z')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(zx_mip, cmap='gray')
plt.title('ZX MIP')
plt.xlabel('X')
plt.ylabel('Z')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(xy_mip, cmap='gray')
plt.title('XY MIP')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()

plt.suptitle(f'MIPs of {deconv_file_path}')
plt.tight_layout()
plt.show()

# %%
