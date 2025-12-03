#%%
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif

mask_path ='/home/confetti/data/rm009/rm009_roi/Z17039_mask.tif'
pred_path ='/home/confetti/data/rm009/rm009_roi/z17030_pred.tif'
roi_path ='/home/confetti/data/rm009/rm009_roi/4/Z17030_C4.tif'

mask = tif.imread(mask_path)
pred = tif.imread(pred_path)
roi = tif.imread(roi_path)
roi_shape = roi.shape
t_s = [int(roi_shape[0]*1/3), int(roi_shape[1]*3/4)]
roi = roi[t_s[0]:,:t_s[1]]
mask= mask[t_s[0]:,:t_s[1]]
pred= pred[t_s[0]:,:t_s[1]]

roi_vmax = np.percentile(roi, 99)
masked_pred = np.ma.masked_where(mask == 0, pred)

plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})
fig, axes = plt.subplots(1, 2, figsize=(20,10))
axes[0].imshow(roi, cmap='gray', vmax=roi_vmax)
axes[0].set_title('Test ROI of V1')
axes[0].axis('off')

axes[1].imshow(roi, cmap='gray', vmax=roi_vmax +500 )
im = axes[1].imshow(masked_pred, cmap='tab10',alpha=0.6, interpolation="nearest")
axes[1].set_title('Prediction')
axes[1].axis('off')

plt.tight_layout()
plt.show()
fig.savefig('./results/pred_v1_whole_slice.png')

# %%
