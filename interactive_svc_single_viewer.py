#!/usr/bin/env python3
"""
Interactive Segmentation UI (v3)
================================
Single-viewer Napari workflow that:
  • Loads a 2D or 3D ROI image
  • Creates a zero-initialized labels layer for sparse user annotations
  • Lets the user choose a segmentation model architecture (cmpsd, DPT, or inception_v3)
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
- DPT uses a DINOv3 backbone + DPTHead (segdino.py). cmpsd uses conv+mlp + ConvSegHead (seg.py). inception_v3 freezes a pretrained Inception backbone and fuses multi-scale features with a lightweight head.

Dependencies
------------
- napari, magicgui, numpy, torch, torchvision (for DINO/DPT utils), scikit-image, zarr (optional)
- Your project modules: lib.arch.seg(ConvSegHead), segdino.DPTHead, lib.arch.ae/build_encoders, helper.one_dim_statis (optional)

"""


from __future__ import annotations
import os
import sys
import numpy as np
import tifffile as tif
import torch
# setting a short tensor print format for easier debugging
torch.set_printoptions(edgeitems=1, threshold=10, linewidth=120)
from typing import Optional

import napari
from time import time
from magicgui import magicgui
 
from lib.arch.segmodel import build_cmpsd, build_dpt, build_and_load_weights_dpt,build_inception_v3, build_seg_head
from lib.utils.preprocess_img import  pad_to_multiple
from helper.napari_view_utilis import find_valid_rectangle_bbox_from_shapes 
from lib.datasets.sparse_label_dataset import SparseLabelSegDataset
from lib.datasets.sparse_label_feats_dataset import SparseLabelFeatsDataset
from lib.datasets.load_rois import load_t1779_1,load_t1779_2, load_3d_rm009

from confettii.plot_helper import three_pca_as_rgb_image
# ----------------------------------
# Project bootstrap (repo root import)
# ----------------------------------
DOWN_FACTOR= 0 
NAPARI = True
PRECOMPUTE_FEAT = False 

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ----------------------------------
# Training/evaluation helpers
# ----------------------------------


from lib.arch.segmodel import Modelsegmodel
from lib.trainers.train_seghead import train_seghead
from lib.inferencers.tilled_inference2d3d import eval_full_roi   

class NapariSegTool:
    def __init__(self, viewer: napari.Viewer|None, dims: int = 3, napari: bool = True,smooth_params=(16,4,1)):
        self.viewer = viewer
        self.dims = dims
        self.napari = napari
        self.smooth_params = smooth_params

        # State Management
        self.roi: Optional[np.ndarray] = None
        self.label: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.segmodel: Optional[Modelsegmodel] = None
        self.pred: Optional[np.ndarray] = None
        self.feat: Optional[np.ndarray] = None
        self.pca_feat: Optional[np.ndarray] = None
        self.offset = (0, 0, 0) if dims == 3 else (0, 0)
        self.final_roi = None
        self.patch_h =0
        self.patch_w = 0
        # UI Config

        self._initialize_data()
        if self.napari:
            self._setup_layers()
        self._init_widgets()


    def _initialize_data(self):
        """Load initial datasets."""
        roi, label, mask = load_t1779_1(region_key='3_3', three_d = (self.dims==3),down_factor=DOWN_FACTOR)
        
        roi_shape = roi.shape[:self.dims]
        
        self.roi = roi
        self.label = label if label is not None else np.zeros(roi_shape, dtype=np.uint8)
        self.mask = mask if mask is not None else np.ones(roi_shape, dtype=bool)
        print(f"{self.roi.shape=}, {self.label.shape=}, {self.mask.shape=}")

    def _setup_layers(self):
        """Prepare Napari layers."""
        for layer_name in ["roi", "user_labels", "mask", "prediction", "feat_rgb"]:
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)
        self.viewer.add_image(self.roi, name="roi")
        self.viewer.add_labels(self.label, name="user_labels")
        self.viewer.add_labels(self.mask, name="mask", opacity=0.3)

    # --- Logic Methods ---
    def perform_training(self, epochs: int, batch_size: int, lr: float, patch_size: tuple):
        """The core training logic."""
        if self.segmodel is None:
            print("Error: Build model first.")
            return
        imagenet_preproc = self.segmodel.name in ["DPT", "inception_v3"] # Simplified check


        if PRECOMPUTE_FEAT:
            ds = SparseLabelFeatsDataset(
                self.roi, self.label, dims=self.dims, 
                patch_size=patch_size, imagenet_preproc=imagenet_preproc
            )
        else:
            ds = SparseLabelSegDataset(
                self.roi, self.label, dims=self.dims, 
                patch_size=patch_size, imagenet_preproc=imagenet_preproc
            )

        n_classes = max(2, len(np.unique(self.label)) - 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        train_seghead(self.segmodel, ds, n_classes=n_classes, device=device,
                      epochs=epochs, batch_size=batch_size, lr=lr, precomute_feat=PRECOMPUTE_FEAT)
        print("Training finished.")


    def _load_train_seghead_to_DPT(self):
        seg_head_state_dict = self.segmodel.seg_model.head.state_dict()
        self.segmodel = build_dpt(dims=self.dims, n_classes=self.segmodel.n_classes, smooth_params=self.smooth_params)
        res = self.segmodel.seg_model.head.load_state_dict(seg_head_state_dict)
        print(f"Loaded seghead weights into DPT: {res}")

    def perform_inference(self, tile_size: tuple, tv_weight: float, capture_features: bool):
        """The core inference/evaluation logic."""
        if self.segmodel is None or self.roi is None:
            print("Error: Model or ROI missing.")
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if PRECOMPUTE_FEAT:
            self._load_train_seghead_to_DPT()
        self.segmodel.seg_model.eval()
        self.segmodel.seg_model.to(device)
        # Handle Bounding Box / ROI selection
        bbox = find_valid_rectangle_bbox_from_shapes({"dims": self.dims, "layers": self.viewer.layers,"roi": self.roi}) if self.napari else None
        
        working_roi = self.roi
        if bbox:
            y0, y1, x0, x1 = bbox
            self.offset = (0, y0, x0) if self.dims == 3 else (y0, x0)
            working_roi = self.roi[y0:y1, x0:x1] if self.dims == 2 else self.roi[:, y0:y1, x0:x1]
        else:
            self.offset = (0,0,0) if self.dims == 3 else (0,0)
        # Inference
        padded_roi = pad_to_multiple(working_roi, 16, dims=self.dims)
        self.final_roi = padded_roi
        
        pred, feat = eval_full_roi(
            self.segmodel, padded_roi, device, 
            tile=tile_size, capture_features=capture_features, 
            tv_denoise_weight=tv_weight
        )
        if bbox:
            if self.dims == 3:
                pred = pred[:,:y1-y0,:x1-x0] #D,H,W
                feat = feat[:,:y1-y0,:x1-x0,:]   if capture_features else None
            else:
                pred = pred[:y1-y0,:x1-x0] #H,W
                feat = feat[:y1-y0,:x1-x0,:]   if capture_features else None

        self.pred, self.feat = pred, feat
        if self.napari:
            self._update_layers(pred, feat)
        
        self.create_pca_rgb()


    def _update_layers(self, pred, feat):
        """Refresh UI layers after inference."""
        if "prediction" in self.viewer.layers:
            self.viewer.layers.remove("prediction")
        self.viewer.add_labels(pred, name="prediction", translate=self.offset)
        
    def create_pca_rgb(self):
        """
        Performs PCA on the feature volume to reduce dimensionality to 3 channels (RGB).
        Uses the mask (if available) to ensure the PCA projection is focused on relevant areas.
        """
        if self.feat is None:
            print("Capture features first (Evaluate with capture_features=True).")
            return

        # 1. Determine spatial dimensions
        # feat is [H, W, C] (2D) or [D, H, W, C] (3D)
        spatial_shape = self.feat.shape[:-1]
        n_channels = self.feat.shape[-1]
        flat_feat = self.feat.reshape(-1, n_channels)

        # 2. Extract the corresponding mask for the current ROI crop
        mask_bool = self._get_mask_for_current_crop(spatial_shape)

        # 3. Calculate PCA
        # If a mask exists, we only compute PCA on pixels inside the mask to 
        # prevent background noise from dominating the color variance.
        if mask_bool is not None and mask_bool.any():
            mask_flat = mask_bool.reshape(-1)
            rgb_flat = np.zeros((mask_flat.size, 3), dtype=np.float32)
            
            # Extract features where mask is True
            masked_features = flat_feat[mask_flat]
            
            # Apply PCA utility (assuming three_pca_as_rgb_image maps features to 0-1 RGB)
            masked_rgb = three_pca_as_rgb_image(masked_features, (int(mask_flat.sum()),))
            
            # Place masked results back into the full flat array
            rgb_flat[mask_flat] = masked_rgb.reshape(-1, 3)
            rgb = rgb_flat.reshape(*spatial_shape, 3)
        else:
            # Fallback to computing on the whole volume
            rgb = three_pca_as_rgb_image(flat_feat, spatial_shape)

        # 4. Add/Update the Napari layer
        if self.napari:
            if "feat_rgb" in self.viewer.layers:
                self.viewer.layers.remove("feat_rgb")

            self.viewer.add_image(
                rgb, 
                name="feat_rgb", 
                rgb=True, 
                blending="additive", 
                translate=self.offset
            )
            print(f"PCA-RGB visualization updated with offset {self.offset}")
        else:
            self.pca_feat = rgb
            print(f"PCA-RGB visualization created.")

    def _get_mask_for_current_crop(self, target_shape) -> Optional[np.ndarray]:
        """
        Helper to slice the global mask to match the current inference window.
        """
        if self.mask is None:
            return None

        mask_view = np.asarray(self.mask)

        # If the mask is already the same size as the features, return it
        if mask_view.shape == target_shape:
            return mask_view.astype(bool)

        # Otherwise, try to crop the mask using the current offset
        try:
            slices = []
            for i in range(len(target_shape)):
                start = int(self.offset[i])
                end = start + target_shape[i]
                slices.append(slice(start, end))
            
            cropped_mask = mask_view[tuple(slices)]
            
            if cropped_mask.shape == target_shape:
                return cropped_mask.astype(bool)
        except Exception as e:
            print(f"Warning: Could not align mask for PCA: {e}")
        
        return None


    def widget_build(self, arch: str = "DPT"):
        classes = np.unique(self.label)
        n_classes = max(2, int(len(classes) - 1))
        
        if arch == "cmpsd":
            self.segmodel = build_cmpsd(dims=self.dims, n_classes=n_classes)
        elif arch == "inception_v3":
            self.segmodel = build_inception_v3(dims=self.dims, n_classes=n_classes)
        elif arch == "DPT":
            if PRECOMPUTE_FEAT:
                self.segmodel = build_seg_head(dims=self.dims, n_classes=n_classes,patch_h=self.patch_h,patch_w=self.patch_w)
            else:
                self.segmodel = build_dpt(dims=self.dims, n_classes=n_classes, smooth_params=self.smooth_params)
                
        else:
            print(f"Error: Unknown architecture '{arch}'.")
            return

        print(f"Built {arch} with {n_classes} classes.")



    @magicgui(
        call_button="Train & Eval",
        arch={"choices": ["cmpsd", "DPT", "inception_v3"]},
        epochs={"min": 1, "max": 50},
        batch_size={"min": 1, "max": 512},
        lr={"step": 1e-4},
        patch_d={"widget_type": "LineEdit"},
        patch_h={"widget_type": "LineEdit"},
        patch_w={"widget_type": "LineEdit"},
    )
    def widget_train_eval(self,
        arch: str = "DPT",
        epochs=15, batch_size=16, lr=1e-4, 
        patch_h=1536, patch_w=1536, patch_d=1,
        # tile_h=1536, tile_w=1536, tile_d=1,
        tv_denoise_weight=0 , capture_features=True
    ):
        #ungly temporary patch_h, pathch_w compute for precompute
        start = time()
        self.patch_h = int(min(int(patch_h),self.roi.shape[0])//16)
        self.patch_w = int(min(int(patch_w),self.roi.shape[1])//16)
        self.widget_build(arch=arch)
        current1 = time()
        print(f"{arch} has been build ,{current1 - start:.2f}s elapsed")

        self.perform_training(epochs, batch_size, lr, (int(patch_d), int(patch_h), int(patch_w)))
        current2 = time()
        print(f"Training completed ,{current2 - current1:.2f}s elapsed")
        
        self.perform_inference((int(patch_d), int(patch_h), int(patch_w)), float(tv_denoise_weight), capture_features)
        print(f"inference time: {time() -current2:.2f}s, Total time: {time()-start:.2f}s")

    

    @magicgui(
        call_button="eval SegHead pretrained",
    )
    def eval_widget_predefined(self,
                    tile_h: int = 1536, tile_w: int = 1536, tile_d: int = 1,
                    tv_denoise_weight : float = 10000,
                    capture_features: bool = True):
        tile_d = int(tile_d)
        tile_h = int(tile_h)
        tile_w = int(tile_w)
        tv_denoise_weight = float(tv_denoise_weight)
        
        #define and load the weights from pretrained seg_model 
        #this pretrained model will predict 8 classes
        self.segmodel = build_and_load_weights_dpt(dims=self.dims)
        self.perform_inference((int(tile_d), int(tile_h), int(tile_w)), float(tv_denoise_weight), capture_features)




    @magicgui(call_button="Enable Double-Click Similarity")
    def widget_similarity(self,):
        @self.viewer.mouse_double_click_callbacks.append
        def on_double_click(_viewer, event):
            if self.feat is None: return
            pos = self.viewer.cursor.position
            # Similarity logic here using self.feat and self.offset
            print(f"Clicked at {pos}")
    # Docking  # --- Widget Definitions ---

    def _init_widgets(self):
        """Define and dock magicgui widgets."""
 
        if self.napari:
            self.viewer.window.add_dock_widget(self.widget_train_eval, area="right", name="1. Train/Eval")
            self.viewer.window.add_dock_widget(self.widget_similarity, area="right", name="2. Analysis")
            self.viewer.window.add_dock_widget(self.eval_widget_predefined, area="right", name="3. Eval Pretrained")

def add_ui(viewer: napari.Viewer,dims,napari=True) -> NapariSegTool:
    # Entry point
    #TODO: automatic dims detection based on ROI shape
    seg_tool = NapariSegTool(viewer,dims,napari=napari)
    return seg_tool


# from napari_orthogonal_views.ortho_view_manager import show_orthogonal_views


def main() -> None:
    os.environ.setdefault("NAPARI_ASYNC", "1")
    dims = 2
    viewer = napari.Viewer(ndisplay=dims)

    # Key binding: toggle predicted segout-like layers
    @viewer.bind_key('v')
    def _toggle_pred(_):
        names = [ln for ln in viewer.layers if any(k in ln.name for k in ("prediction", "segout", "mask", "region", "polygon"))]
        for ln in names:
            viewer.layers[ln].visible = not viewer.layers[ln].visible

    seg_tool = add_ui(viewer,dims=dims,napari=NAPARI)
    # show_orthogonal_views(viewer)
    napari.run()


if __name__ == "__main__":
    main()
