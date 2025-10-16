#!/usr/bin/env python3
"""
Interactive Segmentation UI (v3)
================================
Single-viewer Napari workflow that:
  • Loads a 2D or 3D ROI image
  • Creates a zero-initialized labels layer for sparse user annotations
  • Lets the user choose a segmentation model architecture (cmpsd or DPT)
  • Freezes the backbone and trains only a lightweight seghead from sparse labels
  • Evaluates full-ROI prediction with tiling when needed
  • Captures a feature volume (pre-seghead) for similarity tools
  • Supports dims={2,3} pipelines and per-slice eval for DPT on 3D
  • Preserves double-click: 1D similarity plots (X/Y at current Z) + 3D NCC map

Design notes
-----------
- No dual-viewers and no SimpleViewer widgets. Only one viewer.
- No precomputed featuremap; features are captured from the active backbone.
- Aux/registered masks are intentionally NOT loaded.
- The number of seghead output channels is derived from user label integers.
- DPT uses a DINOv3 backbone + DPTHead (segdino.py). cmpsd uses conv+mlp + ConvSegHead (seg.py).

Dependencies
------------
- napari, magicgui, numpy, torch, torchvision (for DINO/DPT utils), scikit-image, zarr (optional)
- Your project modules: lib.arch.seg(ConvSegHead), segdino.DPTHead, lib.arch.ae/build_encoders, helper.one_dim_statis (optional)

"""


from __future__ import annotations

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import napari
from magicgui import magicgui
from qtpy.QtWidgets import QMessageBox
from confettii.plot_helper import three_pca_as_rgb_image

# ----------------------------------
# Project bootstrap (repo root import)
# ----------------------------------
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ----------------------------------
# Optional / project-local imports
# ----------------------------------
from lib.arch.seg import ConvSegHead  # your lightweight head for cmpsd
from lib.arch.segdino import DPTHead  # your DPT seghead
from config.load_config import load_cfg

# ----------------------------------
# Dataset & utilities
# ----------------------------------

from lib.utils.preprocess_img import  preprocess_uint16_for_imagenet, preprocess_uint8rgb_for_imagenet

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from helper.contrastive_train_helper import class_balance
import tifffile as tif

def load_DKROI():
    rois = []
    for idx in range(270,272):
        roi = tif.imread(f"/home/confetti/data/dk/MD594/MD594/{idx}.tif")
        roi = roi[10000:10000+3200,27000: 27000+ 4800,:]
        h, w = roi.shape[:2]
        trim_h = h % 16
        trim_w = w % 16
        roi = roi[:h -trim_h, :w-trim_w, :]
        rois.append(roi)
    roi = np.array(rois)
    return roi

def load_3d_rm009():
    "the training dataset is from Z55200, Z55500...Z67800 (1um) ,  transfer to 4um space is from Z13800~Z16950"
    "here load a vol seperated from training range"
    vol = tif.imread("/home/confetti/data/rm009/rm009_roi/4/Z13805_C4.tif")
    vol = np.squeeze(vol)
    return vol

def load_t1779():

    mask_vol = tif.imread("/home/confetti/data/t1779/register_data_roi/cp_mask_reduced.tif") 
    mask = mask_vol[5]
    eroded_mask = erode_labels(mask,width=40)
    relabelled_mask,mappings = relabel_sequential(eroded_mask)

    roi_vol = tif.imread("/home/confetti/data/t1779/register_data_roi/cp.tif")
    roi = roi_vol[5]
    return roi, relabelled_mask

# ----------------------------------
# Backbones (project-specific stubs)
# ----------------------------------

import numpy as np
from typing import Optional, Tuple, Dict, Any

def tsne_plot(feats,labels):


    labels = np.array(labels)
    mask = labels > 0
    feats_flat = feats[mask]
    labels_flat = labels[mask]

    blced_feats, blced_labels = class_balance(feats_flat, labels_flat,n_per_class=200)

    tsne_model = TSNE(n_components=2, perplexity=20, random_state=42)
    reduced_data = tsne_model.fit_transform(blced_feats)

    fig,ax = plt.subplots(figsize=(8,8))
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], s=1.2, c=blced_labels, cmap='tab10')
    ax.legend(*scatter.legend_elements(), title="Digits")
    plt.show()



def _round_and_clip_bbox(y0: float, x0: float, y1: float, x1: float, H: int, W: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert float bbox coords to int pixel indices, clipped to [0,H/W], and ensure non-empty.
    Returns (y0, y1, x0, x1) in inclusive-exclusive indexing, or None if invalid.
    """
    y_min = max(0, int(np.floor(min(y0, y1))))
    y_max = min(H, int(np.ceil(max(y0, y1))))
    x_min = max(0, int(np.floor(min(x0, x1))))
    x_max = min(W, int(np.ceil(max(x0, x1))))

    if y_max <= y_min or x_max <= x_min:
        return None
    return (y_min, y_max, x_min, x_max)


def find_valid_rectangle_bbox_from_shapes(state: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    """
    Inspect state['layers']['shapes'] (a napari Shapes layer) and extract a valid rectangle bbox
    in pixel coords relative to the current ROI.
    
    Rules:
      - Uses the last rectangle in the layer that is visible.
      - Accepts napari rectangle data given as 4 corner vertices [[y,x], ...].
      - Validates bbox lies within the current ROI spatial size.

    Returns:
      (y0, y1, x0, x1) in int, inclusive-exclusive indexing; or None if no valid rectangle.

    Assumes:
      - state['roi'] is an array whose first two dims are (H, W) (RGB or grayscale both fine).
      - state['layers']['shapes'] is a napari shapes layer or None.
    """
    
    layers = state.get("layers", {})

    shape_layer_name_set = _filter_layer_name_with_pattern(layers,name_patterns=['Shapes'])
    if len(shape_layer_name_set) == 0:
        return None
    shape_layer_name = next(iter(shape_layer_name_set))

    verts = layers[shape_layer_name].data


    roi = state['roi']
    if roi is None:
        return None
    dims = state['dims']
    if dims ==2:
        H,W = roi.shape[:dims]
    else:
        D,H,W = roi.shape[:dims]
    
    verts = np.asarray(verts)  # expected shape (4, 2): [[y,x], ...]
    verts = np.squeeze(verts)
    ys = verts[:, -2]
    xs = verts[:, -1]
    bbox = _round_and_clip_bbox(ys.min(), xs.min(), ys.max(), xs.max(), H, W)
    if bbox is not None:
        return bbox

    return None


def pad_to_multiple(img: np.ndarray, x: int, dims:int,mode: str = "constant") -> np.ndarray:
    """
    Pad an image so that all dimensions are an integral multiple of x.
    
    Args:
        img (np.ndarray): Input image, shape (H, W) or (D, H, W).
        x (int): The multiple to pad each dimension to.
        mode (str): Padding mode for np.pad (default: "constant").
        **kwargs: Extra arguments passed to np.pad, e.g. constant_values=0.
    
    Returns:
        np.ndarray: Padded image with shape being multiples of x.
    """

    shape = img.shape
    pad_width = []
    
    for dim in shape:
        remainder = dim % x
        if remainder == 0:
            pad_width.append([0,0])
        else:
            pad_width.append([0, x - remainder])
    
    # do not padd the RGB channel dim
    if img.shape[-1] ==3:
        pad_width[-1] = [0,0]

    #do not padd the depth, typical usful for forward in 2d model like dino
    if dims == 3:
        pad_width[0] =[0,0]
    
    padded = np.pad(img, pad_width, mode=mode)
    return padded


def _ensure_tensor_chw_or_cdhw(img: np.ndarray, dims: int,model_name:str) -> torch.Tensor:
    """Convert numpy image to torch tensor with shape (B=1, C=1, H, W) or (B=1, C=1, D, H, W).

    Args:
        img: Input image with shape (H,W) or (D,H,W).
        dims: 2 or 3.
    Returns:
        Torch tensor ready for model input.
    """
    imagenet_preproc = True if model_name =='DPT' else False

    if dims == 2:  

        if imagenet_preproc:              #DPT branch 
            if len(img.shape)==2:
                t = preprocess_uint16_for_imagenet(img)
            else:
                t = preprocess_uint8rgb_for_imagenet(img)


            if t.shape[1] ==1: #DPT model avoid D 
                t= t.squeeze(1)

            t = t.unsqueeze(0) #add B

        else:                             #other model 
            t = torch.from_numpy(img.astype(np.float32))[None, None] #B*C*H*W
    else:
        if imagenet_preproc:
            if len(img.shape) ==3:
                t = preprocess_uint16_for_imagenet(img)
            else:
                t = preprocess_uint8rgb_for_imagenet(img)

            if t.shape[1] ==1:#DPT model avoid D 
                t= t.squeeze(1)
            t = t.unsqueeze(0) #add B

        else:
            t = torch.from_numpy(img.astype(np.float32))[None,None] #B*C*D*H*W

    return t


class SparseLabelSegDataset(Dataset):
    """Dataset from a single ROI and sparse integer labels.

    Logic
    -----
    - For dims=2: use the full image if small, otherwise extract tiles covering labeled pixels.
    - For dims=3: inflate labels along Z and sample tiles around labeled voxels.
    if pathc_size is None, use full image or full slices containing labels.

    Note: This is a scaffold; adapt patch sampling to your data scale.

    Args:
        image: numpy array (H,W) or (D,H,W)
        labels: numpy int array same shape as image
        dims: 2 or 3
        patch_size: tuple for spatial patch (h,w) or (d,h,w)
        max_samples: cap number of sampled patches for quick interaction
    """
    def __init__(self,
                 image: np.ndarray,
                 labels: np.ndarray,
                 dims: int = 2,
                 patch_size: Tuple[int, ...] | None = None,
                 imagenet_preproc: bool = False,
                 max_samples: int = 512):
        super().__init__()
        self.image = image
        self.labels = labels
        self.dims = dims
        self.patch_size = patch_size
        self.samples: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []  # (lo, hi) for slicing
        self.imagenet_preproc = imagenet_preproc

        # the location of all the labeled points
        coords = np.column_stack(np.nonzero(labels))  # N x (2|3)
        if coords.size == 0:
            return


        if dims == 2:
            H, W  = image.shape[:2]
            ph,pw = patch_size
            if (H == ph and W == pw)  or (H < ph)  or  (W < pw):
                # Use full image as one sample if reasonably sized
                self.samples.append(((0, 0), (H, W)))
            else:
                for y, x in coords[::max(1, len(coords)//max_samples)]:
                    y0 = max(0, y - ph//2); y1 = min(H, y0 + ph); y0 = y1 - ph
                    x0 = max(0, x - pw//2); x1 = min(W, x0 + pw); x0 = x1 - pw
                    if y0 < 0 or x0 < 0:
                        continue
                    self.samples.append(((y0, x0), (y1, x1)))
        else:
            D, H, W = image.shape[:3]
            pd, ph, pw = patch_size

            if (pd ==D and ph == H and pw == W) or (pd > D) or (ph > H) or (pw > W): 
                # Use full image as one sample if reasonably sized
                self.samples.append(((0, 0, 0), (D, H, W)))
            else:
                step = max(1, len(coords)//max_samples)
                for z, y, x in coords[::step]:
                    z0 = max(0, z - pd//2); z1 = min(D, z0 + pd); z0 = z1 - pd
                    y0 = max(0, y - ph//2); y1 = min(H, y0 + ph); y0 = y1 - ph
                    x0 = max(0, x - pw//2); x1 = min(W, x0 + pw); x0 = x1 - pw
                    if z0 < 0 or y0 < 0 or x0 < 0:
                        continue
                    self.samples.append(((z0, y0, x0), (z1, y1, x1)))
    


    def __len__(self) -> int:
        return int(len(self.samples))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lo, hi = self.samples[idx]
        if self.dims == 2:
            (y0, x0), (y1, x1) = lo, hi
            img = self.image[y0:y1, x0:x1]
            lab = self.labels[y0:y1, x0:x1]

            if self.imagenet_preproc:
                if len(img.shape) ==2: 
                    x = preprocess_uint16_for_imagenet(img)  # [C,H,W]
                else:
                    x = preprocess_uint8rgb_for_imagenet(img)
            else:
                x = torch.from_numpy(img.astype(np.float32))[None]  # [1,H,W]
            y = torch.from_numpy(lab.astype(np.int64))         # [H,W]
        else:
            (z0, y0, x0), (z1, y1, x1) = lo, hi
            img = self.image[z0:z1, y0:y1, x0:x1]
            lab = self.labels[z0:z1, y0:y1, x0:x1]
            if self.imagenet_preproc:
                if len(img.shape) ==3:
                    x = preprocess_uint16_for_imagenet(img) # [C,D,H,W]
                else:
                    x = preprocess_uint8rgb_for_imagenet(img)
            else:
                x = torch.from_numpy(img.astype(np.float32))[None]  # [1,D,H,W]
            y = torch.from_numpy(lab.astype(np.int64))          # [D,H,W]
        
        # remap label from 0 to N-1
        return x, y - 1


def pad_volume_to_window(volume: np.ndarray, window: Sequence[int]) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Pad a (D,H,W) volume so each dim is a multiple of the window size."""
    if len(window) != 3:
        raise ValueError("Window must have three dimensions for (D,H,W) volumes.")

    D, H, W = volume.shape
    wd, wh, ww = window

    if wd <= 0 or wh <= 0 or ww <= 0:
        raise ValueError("Window dimensions must be positive.")

    pad_d = (wd - (D % wd)) % wd
    pad_h = (wh - (H % wh)) % wh
    pad_w = (ww - (W % ww)) % ww

    if pad_d == pad_h == pad_w == 0:
        return volume, (0, 0, 0)

    padded = np.pad(volume,
                    ((0, pad_d), (0, pad_h), (0, pad_w)),
                    mode="constant", constant_values=0)
    return padded, (pad_d, pad_h, pad_w)

# ----------------------------------
# Model registry & builders
# ----------------------------------

@dataclass
class Modelsegmodel:
    name: str
    dims: int
    seg_model: nn.Module
    n_classes: int


class SimpleSegmodel(nn.Module):
    def __init__(self, encoder: nn.Module, seg_head: nn.Module):
        super().__init__()
        self.cmpsd_encoder = encoder
        self.seg_head = seg_head
        self.feature_map = None  # avoid attribute errors when accessed after eval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 3D case -> [B, C, D, H, W]; 2D case -> [B, C, H, W]
        Returns:
            logits tensor shaped like seg_head output
        """
        features = self.cmpsd_encoder(x)


        out = self.seg_head(features)
        out = F.interpolate(out,tuple(x.shape[2:]),mode='trilinear',align_corners=False)

        # If you only want to cache feature maps during eval:
        if not self.training and hasattr(self, "compute_feature_map"):
            # NOTE: adjust this to whatever shape your compute_feature_map expects
            self.feature_map = self.compute_feature_map(features, x.shape[2:])

        return out

    
    def compute_feature_map(self,features,spatial_shape):
        up = F.interpolate(features,tuple(spatial_shape),mode='trilinear', align_corners=False)
        up= up.squeeze(0).cpu().numpy() # [C,D,H,W]
        up = np.moveaxis(up ,0,-1)  #[D,H,W,C]
        return up
    
    
    def get_feature_map(self):
        if not self.training:
            return self.feature_map
        else:
            return None


def build_cmpsd(dims: int, n_classes: int,) -> Modelsegmodel:
    """Build cmpsd backbone + ConvSegHead.

    Args:
        dims: 2 or 3
        n_classes: number of output classes (incl. background)
        feat_channels: channels produced by backbone
    Returns:
        Modelsegmodel
    """
    level_key = 'l2'
    filters_map={'l1':[32,24,12,12],'l2':[64,32,24,12],'l3':[96,64,32,12]}
    cnn_filters_map ={'l1':[32],'l2':[32,64],'l3':[32,64,96]}
    cnn_kernler_size_map ={'l1':[5],'l2':[5,5],'l3':[5,5,3]}

    from config.load_config import load_cfg
    cfg = load_cfg('/home/confetti/e5_workspace/hive1/outs/contrastive_run_rm009/ae_mlp_rm009_v1/FEATl2_avg8_LOSSpostopk_numparis16384_batch4096_nview4_d_near6_shuffle20_cosdecay_valide_with_avgpool/config.yaml')
    cfg.in_channel = 1
    cfg.filters = cnn_filters_map[level_key] 
    cfg.kernel_size =cnn_kernler_size_map[level_key]
    cfg.mlp_filters = filters_map[level_key]
    cfg.last_encoder =False 
    cfg.avg_pool_size = [8,8,8]

    #todo: try different mlp weights at different epoch: 50, 200, 2000
    from lib.arch.ae_old import build_final_model,load_compose_encoder_dict
    cmpsd_model = build_final_model(cfg)
    cmpsd_model.eval()

    for param in cmpsd_model.parameters():
        param.requires_grad = False

    cnn_ckpt_pth = "/home/confetti/data/weights/ae_feats_nissel_v1_roi1_decaylr_e1600.pth"
    mlp_ckpt_pth = "/home/confetti/e5_workspace/hive1/outs/contrastive_run_rm009/ae_mlp_rm009_v1/FEATl2_avg8_LOSSpostopk_numparis16384_batch4096_nview4_d_near6_shuffle20_cosdecay_valide_with_avgpool/checkpoints/epoch_4000.pth"
    mlp_ckpt = torch.load(mlp_ckpt_pth)['model']
    load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_weight_dict=mlp_ckpt,dims=dims) #this pretrained model is 3d

    from lib.arch.ae import ConvMLP
    seg_head = ConvMLP(filters=[12,12,n_classes],l2_norm=False,last_act=False,dims=dims).train() 
    seg_model = SimpleSegmodel(cmpsd_model,seg_head)


    cnn_ckpt = torch.load(cnn_ckpt_pth)
    print(f"\n\n{cnn_ckpt.keys()}= ")
    print(f"\n\n{mlp_ckpt.keys()}= ")
    print(f"\n\n{seg_model}")
    # summary(seg_model,(1,64,64,64))
     

    print("\n","unfrozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if  p.requires_grad],"\n")

    return Modelsegmodel("cmpsd", 3 ,seg_model,n_classes)


from lib.arch.segdino import DPT,Dinov3HFBackbone
from transformers import AutoModel

def build_dpt(dims: int, n_classes: int, ) -> Modelsegmodel:
    """Build DPT with DINOv3-like backbone + DPTHead.

    Notes:
        - For dims=3, we evaluate per-slice; backbone remains 2D.
        - seghead expects 2D features; for 3D ROI we loop slices externally.
    """

    model_dir = "/home/confetti/e5_workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m"# ViT-S/16 (patch=16)
    hf_backbone = AutoModel.from_pretrained(
        model_dir, local_files_only=True, output_hidden_states=True
    ).eval()
    backbone = Dinov3HFBackbone(hf_backbone)
    seg_model = DPT(nclass=n_classes,backbone=backbone)
    seg_model.train()

    #freeze backbone
    seg_model.lock_backbone()

    print("\n","unfrozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if  p.requires_grad],"\n")

    return Modelsegmodel("DPT", dims,seg_model,n_classes)

def build_and_load_weights_dpt(dims: int, n_classes: int, ) -> Modelsegmodel:
    """Build DPT with DINOv3-like backbone + DPTHead.

    Notes:
        - For dims=3, we evaluate per-slice; backbone remains 2D.
        - seghead expects 2D features; for 3D ROI we loop slices externally.
    """

    model_dir = "/home/confetti/e5_workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m"# ViT-S/16 (patch=16)
    hf_backbone = AutoModel.from_pretrained(
        model_dir, local_files_only=True, output_hidden_states=True
    )
    backbone = Dinov3HFBackbone(hf_backbone)
    seg_model = DPT(nclass=n_classes,backbone=backbone)
    ckpt= torch.load("/home/confetti/e5_workspace/hive1/outs/seg_dino/seg_dino_1zmip/model_epoch_3.pth")
    weights = ckpt['seg_model']

    result = seg_model.load_state_dict(weights)
    print(result)

    #freeze backbone
    seg_model.lock_backbone()
    seg_model.eval()

    print("\n","unfrozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if  p.requires_grad],"\n")

    return Modelsegmodel("DPT", dims,seg_model,n_classes)



# ----------------------------------
# Training/evaluation helpers
# ----------------------------------

def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def one_hot(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """One-hot encode labels to shape [B,C,*]."""
    return F.one_hot(labels.long(), num_classes=n_classes).permute(0, -1, *range(1, labels.dim())).float()


def train_seghead(segmodel: Modelsegmodel,
                  dataset: Dataset,
                  n_classes: int,
                  device: str = "cuda",
                  epochs: int = 5,
                  batch_size: int = 64,
                  lr: float = 1e-3) -> None:
    """Train only the seghead with the backbone frozen.

    Args:
        segmodel: Modelsegmodel (backbone frozen)
        dataset: training dataset from sparse labels
        n_classes: number of classes
        device: 'cuda' or 'cpu'
        epochs: small number for interactivity
        batch_size: mini-batch size
        lr: learning rate
    """
    
    # Freeze backbone should be done in model initialization
    segmodel.seg_model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(segmodel.seg_model.parameters(), lr=lr)
    
    #~~~~~~~ weighted l1 loss ~~~~~~~~#
    from lib.loss.ce_dice_combo import ComboLoss
    from lib.utils.loss_utils import compute_class_weights_from_dataset

    class_weights = compute_class_weights_from_dataset(dataset, num_classes=n_classes,recon_target_flag=False)
    loss_fn = ComboLoss(class_weights=class_weights, focal=True)

    # before training loop (once)
    torch.autograd.set_detect_anomaly(True)
    for n_epoch in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y = y.squeeze(1) 
            if x.shape[2] ==1:
                x = x.squeeze(2)
            if segmodel.name == "DPT" and x.dim() == 5:
                # slice 3D into 2D batches externally
                B, C, D, H, W = x.shape
                x2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
                logits2d = segmodel.seg_model(x2d)  # [B*D, C', H, W]
                logits = logits2d.reshape(B, D, n_classes, H, W).permute(0, 2, 1, 3, 4)
            else:
                logits = segmodel.seg_model(x) #[B,C,D,H,W] or [B,C,H,W]

            # Align logits & labels to same spatial dims
            if logits.dim() == y.dim() + 1:# 2D: logits [B,C,H,W], y [B,H,W] OK# 3D: logits [B,C,D,H,W], y [B,D,H,W] OK
                pass
            else:
                raise RuntimeError("Unexpected logits/labels dims mismatch")
            
            logits_flat = logits.permute(0,*range(2, logits.ndim),1)[ y>= 0]  # [N, K]
            labels_flat = y[ y>= 0]

            # inside the loop, right before backward
            logits_flat = logits_flat.contiguous()
            assert logits_flat.dtype in (torch.float32, torch.float16, torch.bfloat16)
            assert logits_flat.requires_grad, "logits_flat must require grad"
            assert labels_flat.dtype == torch.long, f"labels dtype is {labels_flat.dtype}, expected long"
            assert labels_flat.min().item() >= 0 and labels_flat.max().item() < n_classes, \
                f"label out of range: [{labels_flat.min().item()}, {labels_flat.max().item()}]"
            
            total_loss, ce_loss, dice_loss= loss_fn(logits_flat, labels_flat.long())

            opt.zero_grad(set_to_none=True)
            
            total_loss.backward()  # if it fails, detect_anomaly will print the offending op
            opt.step()

            print(f"epoch:{n_epoch}:train_loss: {total_loss.item():.4f}  "
                    f"(ce={ce_loss.item():.4f}, dice={dice_loss.item():.4f})")

def eval_full_roi(segmodel: Modelsegmodel,
                   image: np.ndarray,
                   device: str = "cuda",
                   tile: Optional[Tuple[int, ...]] = None,
                   capture_features: bool = True,
                   tv_denoise_weight : float = 0.1,
                   ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Evaluate the full ROI, tiling as needed.

    Args:
        segmodel: trained Modelsegmodel
        image: numpy (H,W) or (D,H,W)
        device: torch device
        tile: optional tile size (h,w) or (d,h,w)
        capture_features: whether to return a feature volume (pre-seghead)
    Returns:
        (pred_labels, feature_volume or None)
    """
    dims = segmodel.dims
    n_classes = segmodel.n_classes
    
    # The above line is brittle; better to pass n_classes in real code.
    
    print(f"eval with tv_denoise weight {tv_denoise_weight}")
    with torch.no_grad():
        if dims == 2:
            H, W = image.shape[:dims]
            ph, pw = tile
            if ph == H and pw == W or ph > H or pw >W: 
                x = _ensure_tensor_chw_or_cdhw(image, dims=2,model_name=segmodel.name).to(device)
                logits = segmodel.seg_model(x).squeeze(0) #C*H*W
                probs = F.softmax(logits,dim=0).detach().cpu().numpy()

                if tv_denoise_weight == 0: 
                    pred = np.argmax(probs, axis=0 ) + 1 #C*H*W
                else:
                    denoised_probs = denoise_tv_chambolle(probs, weight=tv_denoise_weight, channel_axis=0)
                    pred = np.argmax(denoised_probs, axis=0) +1

                fvol = segmodel.seg_model.get_feature_map()  if capture_features else None  # [H,W,C]
                return pred, fvol
            else:
                pred = np.zeros((H, W), dtype=np.int32)
                fvol = None
                for y0 in range(0, H, ph):
                    for x0 in range(0, W, pw):
                        y1 = min(H, y0 + ph)
                        x1 = min(W, x0 + pw)
                        tile_img = image[y0:y1, x0:x1]
                        x = _ensure_tensor_chw_or_cdhw(tile_img, 2,model_name=segmodel.name).to(device)
                        logits = segmodel.seg_model(x).squeeze(0) #C*H*W
                        probs = F.softmax(logits,dim=0).detach().cpu().numpy()

                        if tv_denoise_weight == 0: 
                            pred_tile = np.argmax(probs, axis=0 ) + 1 #C*H*W
                        else:
                            denoised_probs = denoise_tv_chambolle(probs, weight=tv_denoise_weight, channel_axis=0)
                            pred_tile = np.argmax(denoised_probs, axis=0) +1

                        pred_tile = np.argmax(probs, axis=0 ) + 1 #C*H*W 
                        pred[y0:y1, x0:x1] = pred_tile

                        if capture_features:
                            f = segmodel.seg_model.get_feature_map() #B,H,W,C
                            if fvol is None:
                                fvol = np.zeros(( H, W,f.shape[-1]), dtype=f.dtype)
                            fvol[ y0:y1, x0:x1,:] = f
                return pred, fvol
        else: # 3d input branch
            D,H, W = image.shape[:dims]
            pd,ph,pw= tile
            pred = np.zeros((D, H, W), dtype=np.int32)
            fvol = None


            if (pd ==D and ph == H and pw == W) or (pd > D) or (ph > H) or (pw > W): 
                # No tiling requested; run a single forward pass
                x = _ensure_tensor_chw_or_cdhw(image, 3, model_name=segmodel.name).to(device)

                if segmodel.name =="DPT" and x.dim() ==5:
                    # slice 3D into 2D batches externally
                    B, C, D, H, W = x.shape
                    x2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
                    logits2d = segmodel.seg_model(x2d)  # [B*D, C', H, W]
                    logits = logits2d.reshape(B, D, n_classes, H, W).permute(0, 2, 1, 3, 4)  #[B, C',D, H, W] 
                else:
                    logits = segmodel.seg_model(x) #[B,C,D,H,W]

                probs = F.softmax(logits, dim=1).detach().cpu().numpy() #[B,C,D,H,W]

                if tv_denoise_weight == 0: 
                    pred = np.argmax(probs, axis=1 ) + 1 #[B,C,D,H,W]
                else:
                    denoised_probs = denoise_tv_chambolle(probs, weight=tv_denoise_weight, channel_axis=1) #[B,C,D,H,W]
                    pred = np.argmax(denoised_probs, axis=1) +1  #[B,C,D,H,W]
                
                pred = np.squeeze(pred) #B equals to 1

                fvol = segmodel.seg_model.get_feature_map()  if capture_features else None  # [B*D,H,W,C] here, B=1

            else:
                # Tiled inference in 3D: tile = (pd, ph, pw)
                pred = np.zeros((D, H, W), dtype=np.int32)
                fvol = None

                for z0 in range(0, D, pd):
                    for y0 in range(0, H, ph):
                        for x0 in range(0, W, pw):
                            z1 = min(D, z0 + pd)
                            y1 = min(H, y0 + ph)
                            x1 = min(W, x0 + pw)

                            tile_img = image[z0:z1, y0:y1, x0:x1]  # [d, h, w] (float/uint)
                            # to tensor [C, D, H, W] and add batch -> [1, C, D, H, W]
                            x = _ensure_tensor_chw_or_cdhw(tile_img, 3, model_name=segmodel.name).to(device)
                            
                            if segmodel.name =="DPT" and x.dim() ==5:
                                # slice 3D into 2D batches externally
                                B, C, D, H, W = x.shape
                                x2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
                                logits2d = segmodel.seg_model(x2d)  # [B*D, C', H, W]
                                logits = logits2d.reshape(B, D, n_classes, H, W).permute(0, 2, 1, 3, 4)  #[B, C',D, H, W] 
                            else:
                                logits = segmodel.seg_model(x) #[B,C,H,W]
                                                
                            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                            pred_tile = np.argmax(probs, axis=1) +1 #[B,C,H,W]
                            pred_tile = np.squeeze(pred_tile) #B equals to 1
                            pred[z0:z1,y0:y1, x0:x1] = pred_tile 

                            if capture_features:
                                f = segmodel.seg_model.get_feature_map() #B*D,H,W,C  (here B equals to 1)
                                if fvol is None:
                                    fvol = np.zeros((D, H, W, f.shape[-1]), dtype=f.dtype)
                                fvol[z0:z1, y0:y1, x0:x1, :] = f
  
            return pred, fvol

# ----------------------------------
# Similarity / NCC utilities
# ----------------------------------
def cosine_similarity_map(anchor: np.ndarray, feat: np.ndarray) -> np.ndarray:
    """Cosine similarity between an anchor feature and every location.

    Args:
        anchor: (C,) feature vector at the clicked location.
        feat:   (H, W, C) or (D, H, W, C) channel-last feature map.

    Returns:
        (H, W) map for 2D, (D, H, W) volume for 3D.
    """
    a = anchor.astype(np.float32)
    a_norm = np.linalg.norm(a) + 1e-8

    if feat.ndim == 3:
        H, W, C = feat.shape
        f = feat.reshape(-1, C).astype(np.float32)      # [HW, C]
        num = f @ a                                     # [HW]
        den = (np.linalg.norm(f, axis=1) * a_norm) + 1e-8
        s = num / den
        return s.reshape(H, W)

    elif feat.ndim == 4:
        D, H, W, C = feat.shape
        f = feat.reshape(-1, C).astype(np.float32)      # [DHW, C]
        num = f @ a
        den = (np.linalg.norm(f, axis=1) * a_norm) + 1e-8
        s = num / den
        return s.reshape(D, H, W)

    else:
        raise ValueError(f"feat.ndim must be 3 or 4, got {feat.ndim}")

def ncc3d(anchor: np.ndarray, feat: np.ndarray) -> np.ndarray:
    """Voxelwise normalized cross-correlation for 3D channel-last features.

    Args:
        anchor: (C,) anchor vector.
        feat:   (D, H, W, C) channel-last feature volume.

    Returns:
        (D, H, W) NCC volume in [-1, 1] (approximately).
    """
    if feat.ndim != 4:
        raise ValueError(f"Expected feat of shape (D,H,W,C), got ndim={feat.ndim}")

    D, H, W, C = feat.shape
    a = anchor.astype(np.float32)
    f = feat.reshape(-1, C).astype(np.float32)          # [DHW, C]

    # Z-normalize across channel dimension
    a = (a - a.mean()) / (a.std() + 1e-8)               # [C]
    f_mean = f.mean(axis=1, keepdims=True)              # [DHW, 1]
    f_std  = f.std(axis=1, keepdims=True) + 1e-8
    f = (f - f_mean) / f_std                            # [DHW, C]

    s = (f * a).sum(axis=1) / C                         # [DHW]
    return s.reshape(D, H, W)   


# ----------------------------------
# Napari UI wiring
# ----------------------------------

def _ask(viewer: napari.Viewer, title: str, text: str) -> None:
    m = QMessageBox(viewer.window._qt_window)
    m.setWindowTitle(title)
    m.setText(text)
    m.exec_()


def add_ui(viewer: napari.Viewer) -> None:
    state = {
        "roi": None,              # np.ndarray (H,W) or (D,H,W)
        "labels": None,           # np.ndarray same shape as roi
        "segmodel": None,           # Modelsegmodel
        "pred": None,             # np.ndarray prediction
        "feat": None,             # np.ndarray feature volume [C,H,W] or [C,D,H,W]
        "dims": 2,
    }

    if "roi" in viewer.layers:
        viewer.layers.remove("roi")
    if "user_labels" in viewer.layers:
        viewer.layers.remove("user_labels")

    # roi = load_3d_rm009()
    # roi = load_DKROI() 
    roi, label = load_t1779()
    roi_shape = roi.shape[:state["dims"]]

    state["roi"] = roi
    state["labels"] = np.zeros(roi_shape,dtype=np.uint8) 
    state["labels"] = label 

    viewer.add_image(state["roi"], name="roi")
    viewer.add_labels(state["labels"], name="user_labels")

    state['layers'] = viewer.layers
    



    # --- Build model button ---
    @magicgui(call_button="Build Model",
              arch={"choices": ["cmpsd", "DPT"]})
    def build_model_widget(arch: str = "DPT"):
        """Instantiate backbone+seghead given current labels for channel count."""
        roi = state["roi"]
        labels = state["labels"]
        if roi is None or labels is None:
            _ask(viewer, "Build Model", "Load ROI first and create labels.")
            return
        classes = np.unique(labels)   
        n_classes = max(2, int(len(classes) -  1))  #ignore the unlabled part 0  ensure >= 2
        dims = state["dims"]
        if arch == "cmpsd":
            state["segmodel"] = build_cmpsd(dims=dims, n_classes=n_classes)
        else:
            state["segmodel"] = build_dpt(dims=dims, n_classes=n_classes)
        _ask(viewer, "Build Model", f"Built {arch} with n_classes={n_classes}, dims={dims}")

    # --- Train seghead ---

    @magicgui(
        call_button="Train SegHead",
        epochs={"min": 1, "max": 50},
        batch_size={"min": 1, "max": 512},
        lr={"step": 1e-4},
        patch_d={"widget_type": "LineEdit"},
        patch_h={"widget_type": "LineEdit"},
        patch_w={"widget_type": "LineEdit"},
    )
    def train_widget(epochs: int = 2, batch_size: int = 16, lr: float = 1e-4,
                     patch_h: int =1024, patch_w: int = 1024,
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
        if segmodel is None:
            _ask(viewer, "Train", "Build the model first.")
            return
        if labels is None or np.count_nonzero(labels) == 0:
            _ask(viewer, "Train", "Please draw some labels on 'user_labels' layer.")
            return
        dims = state["dims"]
        imagenet_preproc = True if segmodel.name =='DPT' else False

        if dims == 2:
            print(f"{roi.shape= }, {labels.shape= }")
            ds = SparseLabelSegDataset(roi, labels, dims=2, patch_size=(patch_h, patch_w),imagenet_preproc=imagenet_preproc)
            # ds = SparseLabelSegDataset(roi, labels, dims=2, patch_size=None, imagenet_preproc=imagenet_preproc)

        else:
            ds = SparseLabelSegDataset(roi, labels, dims=3, patch_size=(patch_d, patch_h, patch_w),imagenet_preproc=imagenet_preproc)
        n_classes = max(2, len(np.unique(labels))-1)
        train_seghead(segmodel, ds, n_classes=n_classes, device="cuda" if torch.cuda.is_available() else "cpu",
                      epochs=epochs, batch_size=batch_size, lr=lr)
        _ask(viewer, "Train", "Training finished.")

    
    
    def _eval_widget(tile_h: int , tile_w: int , tile_d: int,
                     tv_denoise_weight : float,
                     capture_features: bool = False):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        segmodel: Modelsegmodel = state["segmodel"]
        segmodel.seg_model.eval()
        segmodel.seg_model = segmodel.seg_model.to(device)

        roi = state["roi"]

        if segmodel is None or roi is None:
            _ask(viewer, "Evaluate", "Load ROI and build the model first.")
            return

        dims = state["dims"]
        tile = None
        if dims == 2 and (tile_h > 0 and tile_w > 0):
            tile = (tile_h, tile_w)
        elif dims == 3 and (tile_d > 0 and tile_h > 0 and tile_w > 0):
            tile = (tile_d, tile_h, tile_w)
        

        if segmodel.name =="DPT":
            roi = pad_to_multiple(roi, 16,dims=dims)
        
        # Try to discover a valid bbox from the shapes layer
        bbox = find_valid_rectangle_bbox_from_shapes(state)

        if bbox is None:
            pred, feat = eval_full_roi(segmodel, roi, device, tile=tile, capture_features=capture_features,tv_denoise_weight=tv_denoise_weight)
            offset = (0,0) if dims ==2 else (0,0,0) #z,y,x

        else:
            roi_spatial_shape = roi.shape[:dims]
            y0,y1,x0,x1 = bbox
            offset = (y0,x0) if dims ==2 else (0,y0,x0)

            roi_win = roi[y0:y1,x0:x1] if dims==2 else roi[:,y0:y1,x0:x1] 
            padded_roi_win = pad_to_multiple(roi_win,16, dims=dims)

            pred_win, feat_win = eval_full_roi(segmodel, padded_roi_win,device, tile=tile, capture_features=capture_features, tv_denoise_weight=tv_denoise_weight)
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

        # tsne_plot(feat,state["labels"])


        if "prediction" in viewer.layers:
            viewer.layers.remove("prediction")
        viewer.add_labels(pred, name="prediction",translate=offset)
        _ask(viewer, "Evaluate", "Prediction layer added. Feature volume captured." if capture_features else "Prediction done.")

        def _ensure_feat_rgb_layer():
            feat = state.get("feat", None)
            translation = state['offset'] 

            if feat is None:
                _ask(viewer, "PCA-RGB", "Capture features first (Evaluate with capture_features=True).")
                return

            if state["dims"] == 2:
                H, W, C = feat.shape
                rgb = three_pca_as_rgb_image(feat.reshape(-1, C), (H, W))
            else:
                D, H, W, C = feat.shape
                rgb = three_pca_as_rgb_image(feat.reshape(-1, C), (D, H, W))

            if "feat_rgb" in viewer.layers:
                del viewer.layers["feat_rgb"]
            viewer.add_image(rgb, name="feat_rgb", rgb=True, blending="additive",translate=translation)

        # create/update PCA-RGB layer immediately if possible
        # _ensure_feat_rgb_layer()


   # shared widget config (note: you wrote "tw_", assuming you meant "tv_")
    COMMON_WIDGETS = dict(
        tile_d={"widget_type": "LineEdit"},
        tile_h={"widget_type": "LineEdit"},
        tile_w={"widget_type": "LineEdit"},
        tv_denoise_weight={"widget_type": "LineEdit"},
    )
     
    # --- Evaluate ---
    @magicgui(
        call_button="eval SegHead",
        **COMMON_WIDGETS
    )
    def eval_widget(tile_h: int = 512, tile_w: int = 512, tile_d: int = 1,
                   tv_denoise_weight : float = 0.1,
                    capture_features: bool = False):
        tile_d = int(tile_d)
        tile_h = int(tile_h)
        tile_w = int(tile_w)
        tv_denoise_weight = float(tv_denoise_weight)
        _eval_widget(tile_h,tile_w,tile_d,tv_denoise_weight,capture_features=capture_features)
    

    @magicgui(
        call_button="eval SegHead pretrained",
        **COMMON_WIDGETS
    )
    def eval_widget_predefined(tile_h: int = 512, tile_w: int = 512, tile_d: int = 1,
                   tv_denoise_weight : float = 1,
                    capture_features: bool = False):
        tile_d = int(tile_d)
        tile_h = int(tile_h)
        tile_w = int(tile_w)
        tv_denoise_weight = float(tv_denoise_weight)
        
        #define and load the weights from pretrained seg_model 
        state["segmodel"] = build_and_load_weights_dpt(dims=state["dims"],n_classes=8)

        _eval_widget(tile_h,tile_w,tile_d,tv_denoise_weight,capture_features=capture_features)


        
        

    # --- Double-click similarity callbacks ---
    @magicgui(call_button="Enable Double-Click Similarity")
    def enable_similarity():
        """Attach double-click callback: plots 2D cosine-sim map or 3D NCC map.
        
        Also computes and displays a PCA-RGB feature visualization layer ("feat_rgb")
        if `state["feat"]` is available.

        Expects:
            state["feat"]:
                - 2D mode: ndarray [H, W, C]
                - 3D mode: ndarray [D, H, W, C]
            state["dims"]: 2 or 3
        """


        def on_mouse_double_click(_viewer,event):
            if event.type != "mouse_double_click":
                return
            if state.get("feat", None) is None:
                _ask(viewer, "Similarity", "Run Evaluate with 'capture_features=True' first.")
                return

            y0,x0 = state['offset']

            pos = viewer.cursor.position  # (x, y[, z]) in world coords
            
            print(f"double clicked at {pos}")
            if not pos:
                return

            if state['dims'] ==2:
                pos = pos -[x0,y0]

            if state["dims"] == 2:
                x, y = int(round(pos[0])), int(round(pos[1]))
                feat = state["feat"]  # [H, W, C]
                H, W = feat.shape[:2]
                if not (0 <= x < W and 0 <= y < H):
                    return
                anchor = feat[x, y, :].astype(np.float32)  # [C]
                sim_map = cosine_similarity_map(anchor, feat)  # [H, W]
                if "sim2d" in viewer.layers:
                    del viewer.layers["sim2d"]
                viewer.add_image(sim_map, name="sim2d", colormap="magma", blending="additive",translate=[y0,x0])

            else:
                x, y, z = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))
                feat = state["feat"]  # [D, H, W, C]
                D, H, W = feat.shape[:3]
                if not (0 <= x < W and 0 <= y < H and 0 <= z < D):
                    return
                anchor = feat[z, y, x, :].astype(np.float32)  # [C]
                sim3d = ncc3d(anchor, feat)  # [D, H, W]
                if "ncc3d" in viewer.layers:
                    del viewer.layers["ncc3d"]
                viewer.add_image(sim3d, name="ncc3d", colormap="viridis", blending="additive")

        # hotkey placeholder
        viewer.bind_key("D", lambda v: None)
        # similarity callback
        viewer.mouse_double_click_callbacks.append(on_mouse_double_click)


        _ask(viewer, "Similarity", "Double-click similarity enabled. PCA-RGB layer added if features were present.")

    # Dock the widgets
    viewer.window.add_dock_widget(build_model_widget, area="right")
    viewer.window.add_dock_widget(train_widget, area="right")
    viewer.window.add_dock_widget(eval_widget, area="right")
    viewer.window.add_dock_widget(eval_widget_predefined, area="right")
    viewer.window.add_dock_widget(enable_similarity, area="right")


def main() -> None:
    os.environ.setdefault("NAPARI_ASYNC", "1")
    viewer = napari.Viewer(ndisplay=2)

    # Key binding: toggle predicted segout-like layers
    @viewer.bind_key('v')
    def _toggle_pred(_):
        names = [ln for ln in viewer.layers if any(k in ln.name for k in ("prediction", "segout", "mask", "region", "polygon"))]
        for ln in names:
            viewer.layers[ln].visible = not viewer.layers[ln].visible

    add_ui(viewer)
    napari.run()


if __name__ == "__main__":
    main()
