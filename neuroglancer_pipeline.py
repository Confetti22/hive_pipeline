import os
import sys
import numpy as np
import torch
import neuroglancer
import webbrowser
import time

# Project imports (retained from your original code)
from lib.arch.segmodel import build_dpt, build_and_load_weights_dpt
from lib.datasets.sparse_label_dataset import SparseLabelSegDataset
from lib.datasets.load_rois import load_t1779_2
from lib.trainers.train_seghead import train_seghead
from lib.inferencers.tilled_inference2d3d import eval_full_roi
from lib.utils.preprocess_img import pad_to_multiple

class NeuroglancerSegTool:
    def __init__(self, dims: int = 3):
        self.dims = dims
        self.tile =(512,512,512) if dims==3 else (512,512)
        # 1. Start Neuroglancer Server
        self.viewer = neuroglancer.Viewer()
        
        # 2. Load Data
        roi, label, mask = load_t1779_2()
        self.roi = roi.astype(np.float32)
        self.label = label.astype(np.uint32) if label is not None else np.zeros(roi.shape, dtype=np.uint32)
        
        # 3. Setup Coordinate Space
        # Adjust 'voxel_size' based on your actual data resolution (in nm)
        self.dimensions = neuroglancer.CoordinateSpace(
            names=['z', 'y', 'x'] if dims == 3 else ['y', 'x'],
            units='nm',
            scales=[1,1,1] if dims == 3 else [1, 1]
        )

        # 4. Initialize Volumes and Layers
        self.roi_vol = neuroglancer.LocalVolume(data=self.roi, dimensions=self.dimensions)
        
        # Prediction volume initialized as zeros
        self.pred_data = np.zeros_like(self.label,dtype=np.uint8)
        self.pred_vol = neuroglancer.LocalVolume(data=self.pred_data, dimensions=self.dimensions)

        self._setup_viewer_state()
        self._bind_actions()
        
        print(f"Neuroglancer is running at: {self.viewer}")
        webbrowser.open(self.viewer.get_viewer_url())

    def _setup_viewer_state(self):
        """Initializes the browser view with 4 panes (3 ortho + 3D)."""
        with self.viewer.txn() as s:
            s.layers['image'] = neuroglancer.ImageLayer(
                source=self.roi_vol,
                )
            s.layers['user_points'] = neuroglancer.AnnotationLayer()
            s.layers['prediction'] = neuroglancer.SegmentationLayer(source=self.pred_vol)
            s.layout = '4panel' # This provides the 3 orthogonal views + 3D view

    def set_point_tool_callback(self,action_state):
        with self.viewer.txn() as s:
            # Select the annotation layer
            s.selected_layer.layer = 'user_points'
            # Switch to the annotations tab to ensure the tool is visible
            s.layers['user_points'].tab = 'annotations'
            # Note: Since the direct .tool attribute failed, 
            # this ensures the UI focus is correct for manual selection.
            print("Switched to Annotation Layer. Click the '+' icon to start.")

    def _bind_actions(self):
        """Binds keyboard shortcuts to trigger the pipeline."""
        self.viewer.actions.add('train_and_eval', self._trigger_train_eval)
        self.viewer.actions.add('eval_pretrained', self._trigger_pretrained)
        self.viewer.actions.add('point_tool', self.set_point_tool_callback)
        
        with self.viewer.config_state.txn() as s:
            s.input_event_bindings.viewer['keyt'] = 'train_and_eval'
            s.input_event_bindings.viewer['keyp'] = 'eval_pretrained'
            s.input_event_bindings.viewer['keya'] = 'point_tool'
        
        print("Controls: [T] Train & Eval | [P] Eval Pretrained")

    def _get_sparse_mask_from_annotations(self):
        # Create an empty mask
        label = np.zeros(self.roi.shape, dtype=np.uint32)
        
        # Get all annotations from the viewer state
        annotations = self.viewer.state.layers['user_points'].annotations
        
        for ann in annotations:
            # Convert coordinate (floats) to voxel indices (ints)
            # Note: Neuroglancer uses [X, Y, Z] order in its state
            try:
                # Scale coordinates back to array indices
                pos = np.array(ann.point).astype(int)
                z, y, x = pos[2], pos[1], pos[0] # Adjust based on your dims
                
                # Use the 'description' field as the label value
                # If no description, default to label 1
                label_val = int(ann.description) if ann.description.isdigit() else 1
                radius = 6

                if self.dims == 3:
                    z_min = max(0, z - radius)
                    z_max = min(label.shape[0], z + radius + 1)
                    y_min = max(0, y - radius)
                    y_max = min(label.shape[1], y + radius + 1)
                    x_min = max(0, x - radius)
                    x_max = min(label.shape[2], x + radius + 1)

                    zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
                    sphere = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
                    label[z_min:z_max, y_min:y_max, x_min:x_max][sphere] = label_val
                else:
                    y_min = max(0, y - radius)
                    y_max = min(label.shape[0], y + radius + 1)
                    x_min = max(0, x - radius)
                    x_max = min(label.shape[1], x + radius + 1)

                    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                    circle = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
                    label[y_min:y_max, x_min:x_max][circle] = label_val
            except Exception as e:
                print(f"Skipping invalid point: {e}")
                
        return label


    def _trigger_train_eval(self, action_state):
        """Callback for the 'T' key."""
        print("Action Triggered: Training and Inference...")
        
        # 1. Build Model (Default: DPT)
        pixel_label = self._get_sparse_mask_from_annotations() 

        n_classes = max(2, len(np.unique(pixel_label)) - 1)
        self.segmodel = build_dpt(dims=self.dims, n_classes=n_classes)
        
        # 2. Train (Using default values from your magicgui)
        ds = SparseLabelSegDataset(
            self.roi, pixel_label, dims=self.dims, 
            patch_size=(1,512,512), imagenet_preproc=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        train_seghead(self.segmodel, ds, n_classes=n_classes, device=device,
                      epochs=2, batch_size=16, lr=1e-4)
        
        # 3. Inference
        self._run_inference()

    def _trigger_pretrained(self, action_state):
        """Callback for the 'P' key."""
        print("Action Triggered: Loading Pretrained DPT...")
        self.segmodel = build_and_load_weights_dpt(dims=self.dims)
        self._run_inference()

    def _run_inference(self):
        """Core inference logic with volume invalidation for UI refresh."""
        start = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.segmodel.seg_model.eval()
        self.segmodel.seg_model.to(device)

        padded_roi = pad_to_multiple(self.roi, 16, dims=self.dims)
        
        # Run eval
        pred, _ = eval_full_roi(
            self.segmodel, padded_roi, device, 
            tile=self.tile, capture_features=False, tv_denoise_weight=0
        )
        
        # Crop back to original size if padded
        if self.dims == 3:
            self.pred_data[:] = pred[:self.roi.shape[0], :self.roi.shape[1], :self.roi.shape[2]]
        else:
            self.pred_data[:] = pred[:self.roi.shape[0], :self.roi.shape[1]]

        # IMPORTANT: Notify Neuroglancer that the underlying data has changed
        self.pred_vol.invalidate()
        print(f"Inference complete in {time.time()-start:.2f}s. Browser view updated.")

if __name__ == "__main__":
    # Ensure background async is handled if needed
    os.environ.setdefault("NEUROGLANCER_HOSTNAME", "localhost")
    tool = NeuroglancerSegTool(dims=3)
    
    # Keep the script running
    print("Press Ctrl+C to stop the server.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
