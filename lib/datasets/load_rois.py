from helper.image_reader import Ims_Image 
from helper.image_reader import wrap_image
import tifffile as tif
from scipy.ndimage import zoom
import numpy as np

def load_DKROI():
    rois = []
    for idx in range(149,201):
        roi = tif.imread(f"/home/confetti/data/dk/MD594/MD594/{idx}.tif")
        offset = [13200,30000]
        size = [2400,2400]
        roi = roi[offset[0]:offset[0]+size[0],offset[1]:offset[1]+size[1],:]
        roi = zoom(roi,(0.2,0.2,1),order=1)
        h, w = roi.shape[:2]
        trim_h = h % 16
        trim_w = w % 16
        roi = roi[:h -trim_h, :w-trim_w, :]
        rois.append(roi)
    roi = np.array(rois)
    label,mask = None,None
    return roi,label,mask

def load_3d_rm009():
    "the training dataset is from  z55200-z67800 (1um) ,  transfer to 4um space is from Z13800~Z16950"
    "the testing dataset is from  z68100-z69600, transfer to 4um space is Z17025-Z17400 "
    "here load a vol seperated from training range"
    #13750 is the first slice ahead of 55200 and is the 5th in 4um
    vol = tif.imread("/home/confetti/data/rm009/rm009_roi/4/Z13750_C4.tif")
    # vol = tif.imread("/home/confetti/data/rm009/rm009_roi/z16200_z16276C4_d76_h3500_w5250.tif")
    h,w = vol.shape
    print(f"rm009{vol.shape= }")
    # vol = vol[:,:int(h//2),int(w//2):]
    vol = np.squeeze(vol)
    label, mask = None,None
    return vol, label,mask

def get_path_map():
    path_map ={}

    #parent_dir for 'roi' 'mask' and 'gt' is parent1
    parent1='/home/confetti/data/t1779/scenes'
    
    #parent_dir for 'label','pca','pred' is parent2
    parent2 = '/home/confetti/data/t1779/scenes/results'
    #from t1779 dataset
    path_map['1_1'] = {
        'roi': "hp_off7000_2962_4452_sieze1536_1536_12.tif",
        'mask': None,
        'label': 'hp_label.tif',
        'pca':'hp_pca.tif',
        'pred':'hp_pred.tif',
        'pca_incep':'hp_pca_inception_sliding_win.tif',
        'pred_incep':'hp_predict_inception.tif',
        'gt':'1_1_gt.tif',
    }
    path_map['1_2'] = {
        'roi': "vii_1536_1536_83.tif",
        'mask': None,
        'label': '7N_label.tif',
        'pca':'7N_pca.tif',
        'pred':'7N_pred.tif',
        'pca_incep':'7N_pca_inception_sliding_win.tif',
        'pred_incep':'7N_pred_inception.tif',
        'gt':'1_2_gt.tif',
    }
    path_map['1_3'] = {
        'roi': "visa_1536_1536_12.tif",
        'mask': 'visa_mask.tif',
        'label': 'visa_label.tif',
        'pca':'visa_pca.tif',
        'pred':'visa_pred.tif',
        'pca_incep':'visa_pca_inception_sliding_win.tif',
        'pred_incep':'visa_pred_inceptionv3.tif',
        'gt':'1_3_gt.tif',

    }


    # from wide field dataset

    path_map['2_1'] = {
        'roi': "wf_hp_1536_1536.tif",
        'mask': None,
        'label': 'wf_hp_label.tif',
        'pca':'wf_hp_pca.tif',
        'pred':'wf_hp_pred.tif',
        'gt':'2_1_gt.tif',
    }
    path_map['2_2'] = {
        'roi': "wf_viin_1536_1536.tif",
        'mask': None,
        'label': 'wf_7n_label.tif',
        'pca':'wf_7n_pca.tif',
        'pred':'wf_7n_pred.tif',
        'gt':'2_2_gt.tif',
    }
    path_map['2_3']={
        'roi': "wf_visa_1536_1536.tif",
        'mask': 'wf_visa_mask.tif',
        'label': 'wf_visa_label.tif',
        'pca':'wf_visa_pca.tif',
        'pred':'wf_visa_pred.tif',
        'gt':'2_3_gt.tif',
    }
    #from DK dataset

    path_map['3_1'] = {
        'roi': "dk_hp_roi.tif",
        'mask': None,
        'label':'dk_hp_label.tif',
        'pca':'dk_hp_pca.tif',
        'pred':'dk_hp_pred.tif',
        'gt':'3_1_gt.tif',
    }
    path_map['3_2'] = {
        'roi': "dk_vii_roi.tif",
        'mask': None,
        'label':'dk_7N_label.tif',
        'pca':'dk_7N_pca.tif',
        'pred':'dk_7N_pred.tif',
        'gt':'3_2_gt.tif',
    }
    path_map['3_3']={
        'roi': "dk_vis_roi.tif",
        'mask': 'dk_vis_mask.tif',
        'label':'dk_vis_label.tif',
        'pca':'dk_vis_pca.tif',
        'pred':'dk_vis_pred.tif',
        'gt':'3_3_gt.tif',
    }


    return path_map

def load_t1779_2():
    #load a vol around hp
    scale = 2
    o_offset =[7000,2700,3600]
    o_size = [2048,2048,4096]
    offset = [int(coord//2**scale) for coord in o_offset]
    size = [int(sz//2**scale) for sz in o_size]
    image_vol = wrap_image("/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Mouse_Brain/20210131_ZSS_USTC_THY1-YFP_1779_1/Reconstruction_1.0/z00000_c1.ims.part")
    roi = image_vol.from_roi(coords=[*offset, *size],level=scale,channel=2)
    label,mask = None, None
    return roi, label, mask

def load_t1779_3():
    #load a 3d version of '1_3'
    ims_vol =Ims_Image('/home/confetti/e5_data/t1779/t1779.ims',channel=2)
    roi_offset =[6980,3425,4040]
    roi_size =[64,1536,1536]
    roi = ims_vol.from_roi(coords=[*roi_offset,*roi_size],level=0)
    label= np.zeros_like(roi)
    mask = np.zeros_like(roi)
    return roi, label,mask



def load_t1779_1(region_key: str = "2_3",three_d=False,down_factor= 0):
    """
    if thee_d is True, will not using mip

    """

    # mask_vol = tif.imread("/home/confetti/data/t1779/register_data_roi/cp_mask_reduced.tif") 
    # mask = mask_vol[5]
    # eroded_mask = erode_labels(mask,width=70)
    # relabelled_mask,mappings = relabel_sequential(eroded_mask)

    print(f"{region_key= }")
    path_map = get_path_map()

    if region_key not in path_map:
        raise ValueError(f"Unknown region_key '{region_key}'. Available: {list(path_map.keys())}")

    parent_dir = '/home/confetti/data/t1779/scenes'
    roi_path = f"{parent_dir}/{path_map[region_key]['roi']}"
    label_path = f"{parent_dir}/{path_map[region_key]['label']}" if path_map[region_key]['label'] is not None else None
    mask_path = f"{parent_dir}/{path_map[region_key]['mask']}" if path_map[region_key]['mask'] is not None else None

    roi_vol = tif.imread(roi_path)
    roi = np.max(roi_vol,axis=0)  if (len(roi_vol.shape) ==3 and roi_vol.shape[-1] != 3 and not three_d) else roi_vol
    # roi = roi_vol[0]
    roi = np.squeeze(roi)
    
    label = tif.imread(label_path) if label_path is not None else  None
    label = np.squeeze(label) if label is not None else None
    mask = tif.imread(mask_path) if mask_path is not None else None
    mask = np.squeeze(mask) if mask is not None else None   
    if down_factor != 0: 
        zoom_factor = 1 / (2 ** down_factor)
        roi = zoom(roi, zoom=(zoom_factor,zoom_factor,1), order=1)  # downsample to 0.5x for faster testing
        label = zoom(label, zoom=zoom_factor, order=0)  if label is not None else None
        mask = zoom(mask, zoom=zoom_factor, order=0)  if mask is not None else None
    
    #pad the label and mask to be same shape as 3d roi
    if three_d:
        z = roi.shape[0]
        half_z = int(z/2)
        lz = half_z
        rz = half_z if z%2==0 else half_z -1

        if label is not None:
            if label.ndim == 2:
                label = label[None, ...]
            label = np.pad(label, ((lz, rz), (0, 0), (0, 0)), mode="constant", constant_values=0)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, ...]
            mask = np.pad(mask, ((lz, rz), (0, 0), (0, 0)), mode="constant", constant_values=1)

    return roi , label,mask

def load_3d_rm009():
    "the training dataset is from  z55200-z67800 (1um) ,  transfer to 4um space is from Z13800~Z16950"
    "the testing dataset is from  z68100-z69600, transfer to 4um space is Z17025-Z17400 "
    "here load a vol seperated from training range"
    #13750 is the first slice ahead of 55200 and is the 5th in 4um
    vol = tif.imread("/home/confetti/data/rm009/rm009_roi/4/Z13750_C4.tif")
    # vol = tif.imread("/home/confetti/data/rm009/rm009_roi/z16200_z16276C4_d76_h3500_w5250.tif")
    h,w = vol.shape
    print(f"rm009{vol.shape= }")
    # vol = vol[:,:int(h//2),int(w//2):]
    vol = np.squeeze(vol)
    label, mask = None,None
    return vol, label,mask