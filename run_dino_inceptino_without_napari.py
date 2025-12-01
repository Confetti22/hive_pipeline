

import os
import sys
import math
from functools import reduce
from operator import mul
from dataclasses import dataclass
from typing import  List, Optional, Sequence, Tuple, Union
from torchsummary import summary
from skimage.restoration import denoise_tv_chambolle
from helper.napari_view_utilis import _filter_layer_name_with_pattern
from helper.mask_erosion import erode_labels, relabel_sequential
from scipy.ndimage import maximum_filter, zoom

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
from torchvision.transforms import GaussianBlur
from scipy.ndimage import zoom
import tifffile as tif
from interactive_svc_single_viewer import build_dpt,build_inception_v3, build_cmpsd
import napari

from lib.arch.seg import ConvSegHead  # your lightweight head for cmpsd
from lib.arch.segdino import DPTHead  # your DPT seghead
from config.load_config import load_cfg

# ----------------------------------
# Dataset & utilities
# ----------------------------------

from typing import Optional, Tuple, Dict, Any

from lib.utils.preprocess_img import  preprocess_uint16_for_imagenet, preprocess_uint8rgb_for_imagenet

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from helper.contrastive_train_helper import class_balance
import tifffile as tif

from lib.arch.segdino import DPT,Dinov3HFBackbone
from transformers import AutoModel

import numpy as np
import torch
import torch.nn.functional as F
from skimage.restoration import denoise_tv_chambolle
from collections import defaultdict
from typing import Dict, Any, Tuple, Optional
from interactive_svc_single_viewer import  SparseLabelSegDataset, Modelsegmodel, eval_full_roi, train_seghead, _uses_imagenet_preproc, pad_to_multiple, pca_fvol_to_rgb_gpu

DOWNSAMPLE = False 
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

def load_t1779(region_key: str = "2_3"):

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
        dilation_radius = 28
        label= maximum_filter(label, size=2 * dilation_radius + 1, mode="nearest")

    if DOWNSAMPLE is True:
        if len(roi.shape) ==3:
            roi = zoom(roi, zoom=(0.25,0.25,1), order=1)  # downsample to 0.5x for faster testing
        else:
            roi = zoom(roi, zoom=0.25, order=1)  # downsample to 0.5x for faster testing
        label = zoom(label, zoom=0.25, order=0)  if label is not None else None
        mask = zoom(mask, zoom=0.25, order=0)  if mask is not None else None
        gt = zoom(gt, zoom=0.25, order=0)  if gt is not None else None

    return roi , label,mask,gt

def build_model_widget(arch: str = "DPT"):
    """Instantiate backbone+seghead given current labels for channel count."""
    roi = state["roi"]
    labels = state["labels"]
    classes = np.unique(labels)   
    n_classes = max(2, int(len(classes) -  1))  #ignore the unlabled part 0  ensure >= 2
    dims = state["dims"]
    if arch == "cmpsd":
        state["segmodel"] = build_cmpsd(dims=dims, n_classes=n_classes)
    elif arch == "inception_v3":
        state["segmodel"] = build_inception_v3(dims=dims, n_classes=n_classes)
    else:
        state["segmodel"] = build_dpt(dims=dims, n_classes=n_classes)


def train_widget(epochs: int = 50, batch_size: int = 16, lr: float = 1e-4,
                    patch_h: int =1536 , patch_w: int = 1536,
                    patch_d: int = 1):
    """Train lightweight seghead from sparse labels.

    For dims=2, uses (patch_h, patch_w). For dims=3, uses (patch_d, patch_h, patch_w).
    """
        # Force conversion to int
    patch_d = int(patch_d)
    patch_h = int(patch_h)
    patch_w = int(patch_w)
    segmodel: Modelsegmodel = state["segmodel"]
    roi = state["roi"]
    labels = state["labels"]
    dims = state["dims"]
    imagenet_preproc = _uses_imagenet_preproc(segmodel.name)

    if dims == 2:
        print(f"{roi.shape= }, {labels.shape= }")
        ds = SparseLabelSegDataset(roi, labels, dims=2, patch_size=(patch_h, patch_w),imagenet_preproc=imagenet_preproc)
        # ds = SparseLabelSegDataset(roi, labels, dims=2, patch_size=None, imagenet_preproc=imagenet_preproc)

    else:
        ds = SparseLabelSegDataset(roi, labels, dims=3, patch_size=(patch_d, patch_h, patch_w),imagenet_preproc=imagenet_preproc)
    n_classes = max(2, len(np.unique(labels))-1)
    train_seghead(segmodel, ds, n_classes=n_classes, device="cuda" if torch.cuda.is_available() else "cpu",
                    epochs=epochs, batch_size=batch_size, lr=lr)
    # _ask(viewer, "Train", "Training finished.")
    print(f"Training finished.")


def _eval_widget(tile_h: int , tile_w: int , tile_d: int,
                    tv_denoise_weight : float,
                    capture_features: bool = False):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    segmodel: Modelsegmodel = state["segmodel"]
    segmodel.seg_model.eval()
    segmodel.seg_model = segmodel.seg_model.to(device)

    roi = state["roi"]


    dims = state["dims"]
    tile = None
    if dims == 2 and (tile_h > 0 and tile_w > 0):
        tile = (tile_h, tile_w)
    elif dims == 3 and (tile_d > 0 and tile_h > 0 and tile_w > 0):
        tile = (tile_d, tile_h, tile_w)
    

    if segmodel.name =="DPT":
        roi = pad_to_multiple(roi, 16,dims=dims)
    
    # Try to discover a valid bbox from the shapes layer
    bbox = None

    if bbox is None:
        pred, feat = eval_full_roi(segmodel, roi, device, tile=tile, capture_features=capture_features,tv_denoise_weight=tv_denoise_weight)
        offset = (0,0) if dims ==2 else (0,0,0) #z,y,x
        state['final_roi'] = roi

    else:
        roi_spatial_shape = roi.shape[:dims]
        y0,y1,x0,x1 = bbox
        offset = (y0,x0) if dims ==2 else (0,y0,x0)

        roi_win = roi[y0:y1,x0:x1] if dims==2 else roi[:,y0:y1,x0:x1] 
        padded_roi_win = pad_to_multiple(roi_win,16, dims=dims)

        pred_win, feat_win = eval_full_roi(segmodel, padded_roi_win,device, tile=tile, capture_features=capture_features, tv_denoise_weight=tv_denoise_weight)
        state['final_roi'] = padded_roi_win 

        if dims == 3:
            pred_win = pred_win[:,:y1-y0,:x1-x0] #D,H,W
            feat_win = feat_win[:,:y1-y0,:x1-x0]   if capture_features else None
        else:
            pred_win = pred_win[:y1-y0,:x1-x0] #H,W
            feat_win = feat_win[:,:y1-y0,:x1-x0]   if capture_features else None

        pred,feat= pred_win,feat_win


    # pred[pred==1] = 0 #set backgound(1) to 0 to do not display
    state["pred"], state["feat"] = pred, feat
    state['offset'] = offset 
    print(f"eval done, pred shape: {pred.shape}, feat shape: {feat.shape if feat is not None else None}, offset: {offset}")

    # tsne_plot(feat,state["labels"])


    print(f"Prediction layer added. Feature volume captured." if capture_features else "Prediction done.")

    def _ensure_feat_rgb_layer():
        feat = state.get("feat", None)
        translation = state['offset'] 

        mask = state.get("mask", None)
        translation = state.get("offset", (0, 0, 0))

        if state["dims"] == 2:
            H, W, C = feat.shape
            spatial_shape = (H, W)
        else:
            D, H, W, C = feat.shape
            spatial_shape = (D, H, W)

        mask_bool = None
        if mask is not None:
            mask_view = np.asarray(mask)
            if mask_view.shape != spatial_shape:
                try:
                    slices = tuple(
                        slice(int(translation[i]), int(translation[i]) + spatial_shape[i])
                        for i in range(len(spatial_shape))
                    )
                    mask_view = mask_view[slices]
                except Exception:
                    mask_view = None
            if mask_view is not None and mask_view.shape == spatial_shape:
                mask_bool = mask_view.astype(bool)

        flat_feat = feat.reshape(-1, C)
        if mask_bool is None:
            rgb = pca_fvol_to_rgb_gpu(flat_feat, spatial_shape)
        else:
            mask_flat = mask_bool.reshape(-1)
            rgb_flat = np.zeros((mask_flat.size, 3), dtype=np.float32)
            if mask_flat.any():
                masked_rgb = pca_fvol_to_rgb_gpu(flat_feat[mask_flat], (int(mask_flat.sum()),))
                rgb_flat[mask_flat] = masked_rgb.reshape(-1, 3)
            rgb = rgb_flat.reshape(*spatial_shape, 3)


        state['pca'] = rgb
        # create/update PCA-RGB layer immediately if possible

    _ensure_feat_rgb_layer()


def eval_widget(tile_h: int = 1536, tile_w: int = 1536, tile_d: int = 1,
                tv_denoise_weight : float = 100000,
                capture_features: bool = True):
    tile_d = int(tile_d)
    tile_h = int(tile_h)
    tile_w = int(tile_w)
    tv_denoise_weight = float(tv_denoise_weight)
    _eval_widget(tile_h,tile_w,tile_d,tv_denoise_weight,capture_features=capture_features)


def train_and_eval_widget(
                epochs: int = 50, batch_size: int = 16, lr: float = 1e-4,
                patch_h: int =1536 , patch_w: int = 1536,patch_d: int = 1,

                tile_h: int = 1536, tile_w: int = 1536, tile_d: int = 1,
                tv_denoise_weight : float = 100000,
                capture_features: bool = True
                    ):
    train_widget(epochs,batch_size,lr,patch_h,patch_w,patch_d)
    print(f"Training done, starting eval...")
    eval_widget(tile_h,tile_w,tile_d,tv_denoise_weight,capture_features=capture_features)
    print(f"Train+Eval done.")



for arch in ['inception_v3','dpt']:
    # for key in get_path_map().keys():
    for key in ['1_1','2_1','3_1']:
        state = {
        "roi": None,              # np.ndarray (H,W) or (D,H,W)
        "labels": None,           # np.ndarray same shape as roi
        "segmodel": None,           # Modelsegmodel
        "pred": None,             # np.ndarray prediction
        "feat": None,             # np.ndarray feature volume [C,H,W] or [C,D,H,W]
        "dims": 2,
        'pca': None,
        }
        print(f"begin processing region: {key}")
        roi,label,mask,gt= load_t1779(region_key=key)
        roi_shape = roi.shape[:state['dims']]
        dims = state['dims']

        state["roi"] = roi
        state["labels"] = label if label is not None else np.zeros(roi_shape,dtype=np.uint8)
        state["mask"] =  mask if mask is not None else np.ones(roi_shape,dtype=bool)
        state["gt"] =  gt if gt is not None else np.zeros(roi_shape,dtype=np.uint8)


        # #build model

        classes = np.unique(label)   
        n_classes = max(2, int(len(classes) -  1))  #ignore the unlabled part 0  ensure >= 2
        if arch == "cmpsd":
            state["segmodel"] = build_cmpsd(dims=dims, n_classes=n_classes)
        elif arch == "inception_v3":
            state["segmodel"] = build_inception_v3(dims=dims, n_classes=n_classes)
        else:
            state["segmodel"] = build_dpt(dims=dims, n_classes=n_classes)
        train_and_eval_widget(epochs=15)

        save_parent = '/home/confetti/data/t1779/scenes_o'
        
        tif.imwrite(f"{save_parent}/{key}_gt.tif", (state['gt'].astype(np.uint8)))
        tif.imwrite(f"{save_parent}/{key}_roi.tif", (state['roi']).astype(np.uint16))
        tif.imwrite(f"{save_parent}/{key}_label.tif", (state['labels'].astype(np.uint8)))
        tif.imwrite(f"{save_parent}/{key}_mask.tif", state['mask'].astype(np.uint8))

        tif.imwrite(f"{save_parent}/results/{key}_{arch}_pca.tif", (state['pca']*255).astype(np.uint8))
        tif.imwrite(f"{save_parent}/results/{key}_{arch}_pred.tif", state['pred'].astype(np.uint8))
        del state
        

        



    

