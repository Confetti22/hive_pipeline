import os
import sys
from pathlib import Path

import numpy as np
import tifffile as tif
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from config.load_config import load_cfg
from helper.image_reader import Ims_Image
from helper.ncut_helper import segment_and_plot_from_feats
from lib.arch.ae import build_contrastive_model as build_one_stage_model
from lib.utils.preprocess_img import preprocess_uint16_for_imagenet
from lib.inferencers.tilled_inference2d3d import eval_full_roi
from lib.arch.segmodel import Modelsegmodel

class ContrastiveModelWrapper(torch.nn.Module):
    """
    A wrapper to make ContrastiveModel compatible with the tiled inference engine.
    It captures features and ensures spatial resolution matches the tile size.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_features = None
        self.target_size = None
        
    def forward(self, x):
        # x shape: (B, C, D, H, W)
        self.target_size = x.shape[2:] # Store D, H, W for upsampling
        self.last_features = self.model(x)
        # Return features as logits so eval_full_roi can accumulate them.
        return self.last_features
        
    def get_feature_map(self):
        if self.last_features is None:
            return None
            
        # Upsample features back to the input tile size
        upsampled = F.interpolate(
            self.last_features, 
            size=self.target_size, 
            mode='trilinear', 
            align_corners=False
        )
        # Move channel dim to last: (B, C, D, H, W) -> (B, D, H, W, C)
        return np.moveaxis(upsampled.cpu().numpy(), 1, -1)

def load_one_stage_model(device):
    """Load the one-stage contrastive model."""
    config_path = '/home/confetti/e5_workspace/hive1_pipeline/runs/contrastive/onestage_batch2028_nview2_infolossFalse_t1779_2um/config.yaml'
    cfg = load_cfg(config_path)
    cfg.dims = 3
    cfg.avg_pool_size = [4] * 3

    model = build_one_stage_model(cfg)
    model_dir = '/home/confetti/e5_workspace/hive1_pipeline/runs/contrastive/onestage_batch2028_nview2_infolossFalse_t1779_2um/model_epoch_100.pth'
    ckpt = torch.load(model_dir, map_location=device)
    model.load_state_dict(ckpt)
    model.off_proj()
    model.eval().to(device)

    for param in model.parameters():
        param.requires_grad = False
        
    return model

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = '/home/confetti/e5_workspace/hive1_pipeline/results/figs'
    os.makedirs(save_dir, exist_ok=True)

    # 1. Initialization
    print("Loading one-stage model...")
    contrast_one_stage_model = load_one_stage_model(device)

    # 2. Data Loading (Large ROI)
    ims_path = '/home/confetti/e5_data/t1779/t1779.ims'
    z_size, y_size, x_size = 768, 1536, 1536
    
    print(f"Loading large ROI ({z_size}x{y_size}x{x_size}) from {ims_path}...")
    ims_vol = Ims_Image(ims_path, channel=2)
    # Load a representative central ROI
    vol = ims_vol.from_roi([7000,2962,4452, z_size, y_size, x_size])

    # Create 2D RGB visualization for the middle slice
    mid_z = z_size // 2
    slice_2d = vol[mid_z]
    norm_2d = (((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)) * 255).astype(np.uint8)
    rgb_img = np.stack([norm_2d] * 3, axis=-1)

    # Tiled Inference Parameters
    patch_size = (256, 256, 256)
    overlap = 0.25

    # 3. One-Stage Model Tiled Inference
    print("\n--- One-Stage Tiled Inference ---")
    # One-stage model expects 2um resolution (0.5x zoom from 1um)
    vol_zoomed = zoom(vol.astype(np.float32), (0.5, 0.5, 0.5), order=1)
    
    # ImageNet normalization and preprocessing
    processed_vol = preprocess_uint16_for_imagenet(vol_zoomed, make_3ch=False)
    # Remove channel dim for eval_full_roi which expects (D, H, W)
    input_vol_one = processed_vol.squeeze(0).numpy()

    one_stage_wrapped = Modelsegmodel(
        name="one_stage", 
        dims=3, 
        seg_model=ContrastiveModelWrapper(contrast_one_stage_model), 
        n_classes=12
    )

    print(f"Processing zoomed volume {input_vol_one.shape} in patches of {patch_size}...")
    _, one_stage_feats_vol = eval_full_roi(
        one_stage_wrapped,
        input_vol_one,
        device=device,
        tile=patch_size,
        overlap=overlap,
        capture_features=True,
        tv_denoise_weight=0
    )
    
    # Extract features for the visualization slice (zoomed coordinate)
    one_stage_feats_map = one_stage_feats_vol[one_stage_feats_vol.shape[0] // 2]

    # 4. NCut Visualization
    print("\nGenerating NCut visualizations...")
    segment_and_plot_from_feats(
        one_stage_feats_map, 
        rgb_img, 
        n_segments=100, 
        save_dir=save_dir, 
        tag="tiled_large_one_stage"
    )

if __name__ == '__main__':
    main()
