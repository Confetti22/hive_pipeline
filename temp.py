
from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, Optional, Sequence, Tuple
from collections import defaultdict

import napari
from time import time
from magicgui import magicgui
from qtpy.QtWidgets import QMessageBox
 
from skimage.restoration import denoise_tv_chambolle
from helper.napari_view_utilis import _filter_layer_name_with_pattern
from lib.utils.preprocess_img import  preprocess_uint16_for_imagenet, preprocess_uint8rgb_for_imagenet, pad_to_multiple
from lib.datasets.sparse_label_dataset import SparseLabelSegDataset

from confettii.plot_helper import three_pca_as_rgb_image
from confettii.grah_cut import n_cut


def add_ui(viewer: napari.Viewer) -> None:
    state = {
        "roi": None,              # np.ndarray (H,W) or (D,H,W)
        "labels": None,           # np.ndarray same shape as roi
        "segmodel": None,           # Modelsegmodel
        "pred": None,             # np.ndarray prediction
        "feat": None,             # np.ndarray feature volume [C,H,W] or [C,D,H,W]
        "dims": 3,
    }

    if "roi" in viewer.layers:
        viewer.layers.remove("roi")
    if "user_labels" in viewer.layers:
        viewer.layers.remove("user_labels")

    # roi,label,mask = load_3d_rm009()
    # roi,label,mask = load_DKROI() 
    from lib.datasets.load_rois import load_t1779_3
    roi, label,mask = load_t1779_3()
    roi_shape = roi.shape[:state["dims"]]

    state["roi"] = roi
    state["labels"] = label if label is not None else np.zeros(roi_shape,dtype=np.uint8)
    state["mask"] =  mask if mask is not None else np.ones(roi_shape,dtype=bool)

    viewer.add_image(state["roi"], name="roi")
    viewer.add_labels(state["labels"], name="user_labels")
    viewer.add_labels(state["mask"], name="mask",opacity=0.3)

    state['layers'] = viewer.layers
    



    # --- Build model button ---
    from lib.arch.segmodel import build_cmpsd, build_dpt, build_inception_v3
    @magicgui(call_button="Build Model",
              arch={"choices": ["cmpsd", "DPT", "inception_v3"]})
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
        elif arch == "inception_v3":
            state["segmodel"] = build_inception_v3(dims=dims, n_classes=n_classes)
        else:
            state["segmodel"] = build_dpt(dims=dims, n_classes=n_classes)
        _ask(viewer, "Build Model", f"Built {arch} with n_classes={n_classes}, dims={dims}")

    # --- Train seghead ---
   # shared widget config (note: you wrote "tw_", assuming you meant "tv_")
    COMMON_WIDGETS = dict(
        tile_d={"widget_type": "LineEdit"},
        tile_h={"widget_type": "LineEdit"},
        tile_w={"widget_type": "LineEdit"},
        tv_denoise_weight={"widget_type": "LineEdit"},
    )
     


    def train_widget(epochs: int = 20, batch_size: int = 16, lr: float = 1e-4,
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

        if segmodel is None and NAPARI:
            _ask(viewer, "Train", "Build the model first.")
            return
        if (labels is None or np.count_nonzero(labels) == 0) and NAPARI:
            _ask(viewer, "Train", "Please draw some labels on 'user_labels' layer.")
            return

        dims = state["dims"]
        imagenet_preproc = _uses_imagenet_preproc(segmodel.name)

        if dims == 2:
            ds = SparseLabelSegDataset(roi, labels, dims=2, patch_size=(patch_h, patch_w),imagenet_preproc=imagenet_preproc)
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
        if NAPARI:
            bbox = find_valid_rectangle_bbox_from_shapes(state)
        else:
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


        if "prediction" in viewer.layers and NAPARI:
            viewer.layers.remove("prediction")
        viewer.add_labels(pred, name="prediction",translate=offset)
        # _ask(viewer, "Evaluate", "Prediction layer added. Feature volume captured." if capture_features else "Prediction done.")
        print(f"Prediction layer added. Feature volume captured." if capture_features else "Prediction done.")

        def _ensure_feat_rgb_layer():
            feat = state.get("feat", None)
            translation = state['offset'] 

            
            if feat is None and NAPARI:
                _ask(viewer, "PCA-RGB", "Capture features first (Evaluate with capture_features=True).")
                return

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
                rgb = three_pca_as_rgb_image(flat_feat, spatial_shape)
            else:
                mask_flat = mask_bool.reshape(-1)
                rgb_flat = np.zeros((mask_flat.size, 3), dtype=np.float32)
                if mask_flat.any():
                    masked_rgb = three_pca_as_rgb_image(flat_feat[mask_flat], (int(mask_flat.sum()),))
                    rgb_flat[mask_flat] = masked_rgb.reshape(-1, 3)
                rgb = rgb_flat.reshape(*spatial_shape, 3)

            if "feat_rgb" in viewer.layers and NAPARI:
                del viewer.layers["feat_rgb"]

            if NAPARI:
                viewer.add_image(rgb, name="feat_rgb", rgb=True, blending="additive",translate=translation)
            state['pca'] = rgb
        
        def _ncut():
            feat = state.get("feat", None)
            translation = state['offset'] 
            mask = state.get("mask", None)

            if feat is None:
                _ask(viewer, "NCut", "Capture features first (Evaluate with capture_features=True).")
                return

            mask_bool = None
            if mask is not None:
                mask_view = np.asarray(mask)
                spatial_shape = feat.shape[:2] if state["dims"] == 2 else feat.shape[:3]
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

            if state["dims"] == 2:
                H, W, C = feat.shape
                input_image = state['final_roi'] 
                if mask_bool is not None and mask_bool.any():
                    coords = np.argwhere(mask_bool)
                    min_idx = coords.min(axis=0)
                    max_idx = coords.max(axis=0) + 1
                    region_slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(min_idx, max_idx))
                    mask_crop = mask_bool[region_slices]
                    feat_crop = feat[region_slices + (slice(None),)]
                    feat_crop = np.where(mask_crop[..., None], feat_crop, 0)
                    input_crop = input_image[region_slices]
                    ncut_crop = n_cut(feat_crop, input_crop)
                    ncut_map = np.zeros((H, W), dtype=ncut_crop.dtype)
                    ncut_view = ncut_map[region_slices]
                    ncut_view[mask_crop] = ncut_crop[mask_crop]
                else:
                    ncut_map = n_cut(feat,input_image)

            if "ncut" in viewer.layers:
                del viewer.layers["ncut"]
            viewer.add_image(ncut_map, name="ncut", colormap="viridis", blending="additive",translate=translation)

        # create/update PCA-RGB layer immediately if possible
        _ensure_feat_rgb_layer()
        # _ncut()

    def eval_widget(tile_h: int = 1536, tile_w: int = 1536, tile_d: int = 1,
                   tv_denoise_weight : float = 100000,
                    capture_features: bool = True):
        tile_d = int(tile_d)
        tile_h = int(tile_h)
        tile_w = int(tile_w)
        tv_denoise_weight = float(tv_denoise_weight)
        _eval_widget(tile_h,tile_w,tile_d,tv_denoise_weight,capture_features=capture_features)


    @magicgui(
    call_button="Train SegHead and then eval",
    epochs={"min": 1, "max": 50},
    batch_size={"min": 1, "max": 512},
    lr={"step": 1e-4},
    patch_d={"widget_type": "LineEdit"},
    patch_h={"widget_type": "LineEdit"},
    patch_w={"widget_type": "LineEdit"},
    ** COMMON_WIDGETS
    )
    def train_and_eval_widget(
                    epochs: int = 2, batch_size: int = 16, lr: float = 1e-4,
                    patch_h: int =512 , patch_w: int = 512,patch_d: int = 1,

                    tile_h: int = 512, tile_w: int = 512, tile_d: int = 1,
                    tv_denoise_weight : float = 100000,
                    capture_features: bool = True
                     ):
        begin = time()
        train_widget(epochs,batch_size,lr,patch_h,patch_w,patch_d)
        current = time()
        print(f"Training done, train time{current - begin:.4f}")
        eval_widget(tile_h,tile_w,tile_d,tv_denoise_weight,capture_features=capture_features)
        end = time()
        print(f"Train+Eval done. eval time{end -current:.4f}, total time{end - begin:.4f}")
        

    from lib.arch.segmodel import build_and_load_weights_dpt
    @magicgui(
        call_button="eval SegHead pretrained",
        **COMMON_WIDGETS
    )
    def eval_widget_predefined(tile_h: int = 512, tile_w: int = 512, tile_d: int = 1,
                   tv_denoise_weight : float = 10000,
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
    viewer.window.add_dock_widget(train_and_eval_widget, area="right")
    viewer.window.add_dock_widget(eval_widget_predefined, area="right")
    viewer.window.add_dock_widget(enable_similarity, area="right")



