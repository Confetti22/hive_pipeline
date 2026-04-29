
import tifffile as tif
import numpy as np
from scipy.ndimage import zoom

def get_path_map():
    path_map ={}


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
        'mask': "7N_mask.tif",
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
        'mask': "dk_7N_mask.tif",
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


def get_path_map_z64():
    """
    hp, vii are z64 version of get_path_map; visa2 is a different roi at visa
    """
    
    path_map ={}

    #from t1779 dataset
    path_map['1_1'] = {
        'roi': "hp_1536_1536_64.tif",
        'mask': None,
        'label': 'hp_label.tif',
        'pca':'hp_pca.tif',
        'pred':'hp_pred.tif',
        'pca_incep':'hp_pca_inception_sliding_win.tif',
        'pred_incep':'hp_predict_inception.tif',
        'gt':'1_1_gt.tif',
    }
    path_map['1_2'] = {
        'roi': "vii_1536_1536_64.tif",
        'mask': "7N_mask.tif",
        'label': '7N_label.tif',
        'pca':'7N_pca.tif',
        'pred':'7N_pred.tif',
        'pca_incep':'7N_pca_inception_sliding_win.tif',
        'pred_incep':'7N_pred_inception.tif',
        'gt':'1_2_gt.tif',
    }

    path_map['1_3'] = {
        # visa region roi2 
        'roi': "visa2_1536_1536_64.tif",
        'mask': 'visa2_mask.tif',
        'label': 'visa2_sparse_label.tif',
        'pca':'visa_pca.tif',
        'pred':'visa_pred.tif',
        'pca_incep':'visa_pca_inception_sliding_win.tif',
        'pred_incep':'visa_pred_inceptionv3.tif',
        'gt':'visa2_gt.tif',

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
        'mask': "dk_7N_mask.tif",
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

def load_t1779(region_key: str = "2_3", DOWNSAMPLE = True ):

    #parent_dir for 'roi' 'mask' and 'gt' is parent1
    parent1='/home/confetti/data/t1779/scenes'
    
    #parent_dir for 'label','pca','pred' is parent2
    parent2 = '/home/confetti/data/t1779/scenes/results'# relabelled_mask,mappings = relabel_sequential(eroded_mask)

    print(f"{region_key= }")
    path_map = get_path_map()

    if region_key not in path_map:
        raise ValueError(f"Unknown region_key '{region_key}'. Available: {list(path_map.keys())}")

    roi_path = f"{parent1}/{path_map[region_key]['roi']}"
    label_path = f"{parent1}/{path_map[region_key]['label']}" if path_map[region_key]['label'] is not None else None
    mask_path = f"{parent1}/{path_map[region_key]['mask']}" if path_map[region_key]['mask'] is not None else None
    gt_path = f"{parent1}/{path_map[region_key]['gt']}" if path_map[region_key]['gt'] is not None else None

    roi_vol = tif.imread(roi_path)
    roi = np.max(roi_vol,axis=0)  if (len(roi_vol.shape) ==3 and roi_vol.shape[-1] != 3) else roi_vol
    # roi = roi_vol[0]
    roi = np.squeeze(roi)
    
    label = tif.imread(label_path) if label_path is not None else  None
    label = np.squeeze(label) if label is not None else None
    mask = tif.imread(mask_path) if mask_path is not None else None
    mask = np.squeeze(mask) if mask is not None else None   
    gt = tif.imread(gt_path) if gt_path is not None else None
    gt = np.squeeze(gt) if gt is not None else None

    if region_key in ['1_1','2_1','3_1','1_2','1_3']:
        
        from skimage.segmentation import expand_labels
        dilation_radius = 8
        label = expand_labels(label,dilation_radius)

    if DOWNSAMPLE is True:
        if len(roi.shape) ==3:
            roi = zoom(roi, zoom=(0.25,0.25,1), order=1)  # downsample to 0.5x for faster testing
        else:
            roi = zoom(roi, zoom=0.25, order=1)  # downsample to 0.5x for faster testing
        label = zoom(label, zoom=0.25, order=0)  if label is not None else None
        mask = zoom(mask, zoom=0.25, order=0)  if mask is not None else None
        gt = zoom(gt, zoom=0.25, order=0)  if gt is not None else None

    return roi , label,mask,gt