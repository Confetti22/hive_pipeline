#%%
"""



Unified Segmentation Testing Interface

This script provides an interactive interface for testing different segmentation methods
on multiple datasets (t1779 and rm009). It supports various segmentation algorithms
including graph cut, MLP-based, CNN-based, and similarity-based methods.

Author: AI Assistant
Date: 2024
"""

import sys
import os
import time
from typing import Dict, Tuple, Optional, Any

# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

import torch
import numpy as np
import napari
from magicgui import widgets
from magicgui.widgets import Container
from scipy.ndimage import zoom
import tifffile as tif
from sklearn.decomposition import PCA
from transformers import AutoModel

# Local imports
from helper.graph_cut_helper import GraphCutFastBFS
from lib.arch.ae_old import (
    build_final_model, load_compose_encoder_dict, 
    build_encoder_model, load_encoder2encoder
)
from config.load_config import load_cfg
from helper.image_seger import _compute_seg2, _seg_via_conv_head, _seg_via_mlp_head
from lib.utils.preprocess_img import pad_to_multiple_of_unit
from helper.image_reader import Ims_Image

# Grow-cut segmentation import


class DatasetConfig:
    """Configuration class for dataset paths and parameters."""
    
    CONFIGS = {
        't1779': {
            'ims_path': '/home/confetti/e5_data/t1779/t1779.ims',
            'roi_offset': [6980, 3425, 4040],
            'roi_size': [64, 1536, 1536],
            'test_data_path': '/home/confetti/data/t1779/test_data_part_brain/0001.tif',
            'user_input_path': '/home/confetti/data/t1779/test_data_part_brain/0001_user_input.tif'
        },
        'rm009': {
            'ims_path': '/home/confetti/e5_data/rm009/rm009.ims',
            'roi_offset': [6980, 3425, 4040],
            'roi_size': [64, 1536, 1536],
            'test_data_path': '/home/confetti/data/rm009/test_data_part_brain/0001.tif',
            'user_input_path': '/home/confetti/data/rm009/test_data_part_brain/0001_user_input.tif'
        }
    }
    
    @classmethod
    def get_config(cls, dataset_name: str) -> Dict[str, Any]:
        """Get configuration for a specific dataset."""
        if dataset_name not in cls.CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cls.CONFIGS.keys())}")
        return cls.CONFIGS[dataset_name]
    
    @classmethod
    def get_available_datasets(cls) -> list:
        """Get list of available datasets."""
        return list(cls.CONFIGS.keys())


class DINOv3FeatureExtractor:
    """DINOv3 feature extractor for extracting multi-layer features."""
    
    def __init__(self, model_dir: str = "/home/confetti/e5_workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m",
                 device: str = 'cuda', patch_size: int = 16):
        """
        Initialize DINOv3 feature extractor.
        
        Args:
            model_dir: Path to the DINOv3 model directory
            device: Device to run the model on
            patch_size: Patch size for the model (16 for ViT-S/16)
        """
        self.device = device
        self.patch_size = patch_size
        self.model = None
        self._load_model(model_dir)
    
    def _load_model(self, model_dir: str):
        """Load the DINOv3 model from HuggingFace."""
        print(f"Loading DINOv3 model from {model_dir}...")
        self.model = AutoModel.from_pretrained(
            model_dir, local_files_only=True, output_hidden_states=True
        )
        self.model.eval().to(self.device)
        print("DINOv3 model loaded successfully!")
    
    def extract_features(self, image: np.ndarray, layer_indices: list = [2, 5, 8, 11],
                        pca_dim: int = None) -> np.ndarray:
        """
        Extract features from specified layers of DINOv3.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            layer_indices: List of layer indices to extract features from
            pca_dim: Optional PCA dimensionality reduction (None to skip)
            
        Returns:
            Feature map as numpy array (H, W, C)
        """
        # Ensure image is 2D grayscale
        if len(image.shape) == 3:
            image = image.mean(axis=2)
        
        # Convert to tensor and add batch and channel dimensions
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Convert to 3-channel RGB by repeating the grayscale image
        image_tensor = image_tensor.repeat(1, 3, 1, 1)  # (1, 3, H, W)
        
        # Normalize to ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        image_tensor = image_tensor.to(self.device)
        
        # Get patch dimensions
        patch_h = image_tensor.shape[-2] // self.patch_size
        patch_w = image_tensor.shape[-1] // self.patch_size
        
        # Extract features from specified layers
        with torch.no_grad():
            outputs = self.model(pixel_values=image_tensor, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            
            # Extract features from specified layers (skip first 5 tokens: 1 CLS + 4 REGISTER)
            layer_features = []
            for layer_idx in layer_indices:
                tokens = hidden_states[layer_idx + 1]  # [B, 5+HW, C]
                patch_tokens = tokens[:, 5:, :]  # Drop CLS + 4 REGISTER -> [B, HW, C]
                layer_features.append(patch_tokens)
        
        # Reshape tokens from B*N*C to B*H*W*C
        reshaped_features = []
        for feat in layer_features:
            # feat shape: [B, HW, C]
            B, HW, C = feat.shape
            feat_reshaped = feat.reshape(B, patch_h, patch_w, C)  # [B, H, W, C]
            reshaped_features.append(feat_reshaped)
        
        # Concatenate all layer features along channel dimension
        fused_features = torch.cat(reshaped_features, dim=-1)  # [B, H, W, C_total]
        
        # Apply PCA if requested
        if pca_dim is not None:
            # Reshape for PCA: (B*H*W, C_total)
            B, H, W, C = fused_features.shape
            feat_flat = fused_features.reshape(B * H * W, C).cpu().numpy()
            
            # Apply PCA
            pca = PCA(n_components=pca_dim)
            feat_pca = pca.fit_transform(feat_flat)  # (B*H*W, pca_dim)
            
            # Reshape back to spatial dimensions
            fused_features = torch.from_numpy(feat_pca).reshape(B, H, W, pca_dim)
        
        # Convert to numpy and remove batch dimension
        fused_features = fused_features.squeeze(0).cpu().numpy()  # (H, W, C)
        
        # Upsample to original image size
        if fused_features.shape[:2] != image.shape:
            # Use bilinear interpolation to upsample
            fused_tensor = torch.from_numpy(fused_features).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            upsampled = torch.nn.functional.interpolate(
                fused_tensor, size=image.shape, mode='bilinear', align_corners=False
            )
            fused_features = upsampled.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C)
        
        return fused_features


class ModelManager:
    """Manages the neural network models and their loading."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.cmpsd_model = None
        self.encoder_model = None
        self.dinov3_extractor = None
        self._load_models()
    
    def _load_models(self):
        """Load and initialize the neural network models."""
        print("Loading models...")
        
        # Load configuration
        args = load_cfg('config/t11_3d.yaml')
        args.avg_pool_size = (8, 8, 8)
        
        # Build and load composite model
        self.cmpsd_model = build_final_model(args)
        self.cmpsd_model.eval().to(self.device)
        
        # Load trained weights
        cnn_ckpt_pth = '/home/confetti/data/weights/t11_3d_ae_best2.pth'
        mlp_ckpt_pth = '/home/confetti/data/weights/t11_3d_mlp_best_new_format.pth'
        load_compose_encoder_dict(self.cmpsd_model, cnn_ckpt_pth, mlp_ckpt_pth, dims=args.dims)
        
        # Build and load encoder model
        self.encoder_model = build_encoder_model(args, dims=3)
        self.encoder_model.eval().to(self.device)
        load_encoder2encoder(self.encoder_model, cnn_ckpt_pth)
        
        # Load DINOv3 feature extractor
        self.dinov3_extractor = DINOv3FeatureExtractor(device=self.device)
        
        print("Models loaded successfully!")


class DataManager:
    """Manages data loading and feature computation."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.current_dataset = 't1779'
        self.vol = None
        self.mlp_out = None
        self.feats_map = None
        self.rgb_vis = None
        self.dinov3_feats = None
        self.dinov3_rgb_vis = None
        self.z_slice = None
        self.input_label_shape = None
        
        # Load initial dataset
        self.load_dataset(self.current_dataset)
    
    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple]:
        """Load dataset and compute features."""
        print(f"Loading dataset: {dataset_name}")
        
        config = DatasetConfig.get_config(dataset_name)
        self.current_dataset = dataset_name
        
        # Load data from IMS file
        ims_vol = Ims_Image(config['ims_path'], channel=2)
        roi_offset = config['roi_offset']
        roi_size = config['roi_size']
        vol = ims_vol.from_roi(coords=[*roi_offset, *roi_size], level=0)
        
        # Load test data
        vol = tif.imread(config['test_data_path'])
        zoom_factor = 8
        print(f"Original volume shape: {vol.shape}")
        
        # Pad volume to multiple of zoom factor
        vol = pad_to_multiple_of_unit(vol, unit=zoom_factor)
        print(f"Padded volume shape: {vol.shape}")
        
        # Compute features using the model
        input_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(self.model_manager.device)
        mlp_out = self.model_manager.cmpsd_model(input_tensor).cpu().detach().squeeze().numpy()
        print(f"Feature map shape: {mlp_out.shape}")
        
        C, H, W = mlp_out.shape
        
        # Process feature map
        feats_map = mlp_out[:, :, :]
        feats_map = np.moveaxis(feats_map, 0, -1)  # Shape: (H, W, C)
        
        # Compute PCA visualization
        pca = PCA(n_components=3)
        rgb_vis = pca.fit_transform(feats_map.reshape(-1, C)).reshape(H, W, 3)
        
        # Compute DINOv3 features for the z-slice
        print("Computing DINOv3 features...")
        dinov3_feats = self.model_manager.dinov3_extractor.extract_features(
            self.z_slice, layer_indices=[2, 5, 8, 11], pca_dim=256
        )
        
        # Compute DINOv3 PCA visualization
        dinov3_pca = PCA(n_components=3)
        dinov3_rgb_vis = dinov3_pca.fit_transform(dinov3_feats.reshape(-1, dinov3_feats.shape[-1])).reshape(dinov3_feats.shape[0], dinov3_feats.shape[1], 3)
        
        # Store results
        self.vol = vol
        self.mlp_out = mlp_out
        self.feats_map = feats_map
        self.rgb_vis = rgb_vis
        self.dinov3_feats = dinov3_feats
        self.dinov3_rgb_vis = dinov3_rgb_vis
        self.z_slice = vol[32]
        self.input_label_shape = self.z_slice.shape
        
        print(f"Dataset {dataset_name} loaded successfully!")
        return vol, mlp_out, feats_map, rgb_vis, self.z_slice, self.input_label_shape
    
    def get_label_data(self) -> np.ndarray:
        """Load user input labels for current dataset."""
        config = DatasetConfig.get_config(self.current_dataset)
        label_data = tif.imread(config['user_input_path'])
        return label_data.astype(int)
    
    def recompute_dinov3_features(self, layer_indices: list = [2, 5, 8, 11], pca_dim: int = 256):
        """Recompute DINOv3 features with new parameters."""
        print("Recomputing DINOv3 features...")
        
        # Compute DINOv3 features for the z-slice
        dinov3_feats = self.model_manager.dinov3_extractor.extract_features(
            self.z_slice, layer_indices=layer_indices, pca_dim=pca_dim
        )
        
        # Compute DINOv3 PCA visualization
        dinov3_pca = PCA(n_components=3)
        dinov3_rgb_vis = dinov3_pca.fit_transform(dinov3_feats.reshape(-1, dinov3_feats.shape[-1])).reshape(dinov3_feats.shape[0], dinov3_feats.shape[1], 3)
        
        # Update stored features
        self.dinov3_feats = dinov3_feats
        self.dinov3_rgb_vis = dinov3_rgb_vis
        
        print("DINOv3 features recomputed successfully!")


class SegmentationMethods:
    """Contains all segmentation method implementations."""
    
    @staticmethod
    def similarity_segmentation(label_mask: np.ndarray, feature_map: np.ndarray, 
                              spatial_decay: bool = True) -> np.ndarray:
        """
        Similarity-based segmentation using feature similarity.
        
        Args:
            label_mask: Input label mask
            feature_map: Feature map for similarity computation
            spatial_decay: Whether to use spatial decay
            
        Returns:
            Segmentation result
        """
        feats_map_shape = feature_map.shape[:-1]
        input_label_shape = label_mask.shape
        
        # Downscale label mask to feature map resolution
        zoom_factors = [x/y for x, y in zip(feats_map_shape, input_label_shape)]
        label_mask = zoom(label_mask, zoom=zoom_factors, order=0)
        print(f"Downscaled label mask shape: {label_mask.shape}")
        
        # Prepare inputs for segmentation
        feature_map = np.expand_dims(feature_map, axis=0)
        label_mask = np.expand_dims(label_mask, axis=0)
        
        # Compute segmentation
        result = _compute_seg2(label_mask, feature_map, spatial_decay)
        result = np.squeeze(result)
        
        # Upscale result back to original resolution
        zoom_factors = [y/x for x, y in zip(feats_map_shape, input_label_shape)]
        zoomed_seg_label = zoom(result, zoom=zoom_factors, order=0)
        
        return zoomed_seg_label.astype(np.uint8)
    
    @staticmethod
    def graph_cut_segmentation(label_mask: np.ndarray, feats_map: np.ndarray,
                              sigma: float, lambda_val: float) -> np.ndarray:
        """
        Graph cut-based segmentation.
        
        Args:
            label_mask: Input label mask
            feats_map: Feature map
            sigma: Graph cut sigma parameter
            lambda_val: Graph cut lambda parameter
            
        Returns:
            Segmentation result
        """
        feats_map_shape = feats_map.shape[:-1]
        input_label_shape = label_mask.shape
        
        # Downscale label mask
        zoom_factors = [x/y for x, y in zip(feats_map_shape, input_label_shape)]
        label_mask = zoom(label_mask, zoom=zoom_factors, order=0)
        print(f"Downscaled label mask shape: {label_mask.shape}")
        
        # Check for valid labels
        unique_labels = np.unique(label_mask)
        unique_labels = unique_labels[unique_labels != 0]  # Ignore background
        
        if len(unique_labels) < 2:
            print("Warning: Not enough labels for graph cut")
            return np.zeros(label_mask.shape, dtype=np.uint8)
        
        # Perform graph cut
        graph_cut = GraphCutFastBFS(feats_map, label_mask, sigma, lambda_val)
        graph_cut.start_cut()
        result = graph_cut.TREE
        
        # Upscale result
        zoom_factors = [y/x for x, y in zip(feats_map_shape, input_label_shape)]
        zoomed_seg_label = zoom(result, zoom=zoom_factors, order=0)
        
        return zoomed_seg_label.astype(np.uint8)
    
    @staticmethod
    def head_based_segmentation(label_mask: np.ndarray, feats_map: np.ndarray,
                              num_epochs: int = 2000, mode: str = 'mlp',
                              return_prob: bool = False) -> np.ndarray:
        """
        Segmentation using MLP or CNN heads.
        
        Args:
            label_mask: Input label mask
            feats_map: Feature map
            num_epochs: Number of training epochs
            mode: 'mlp' or 'conv'
            return_prob: Whether to return probability maps
            
        Returns:
            Segmentation result or probability maps
        """
        feats_map_shape = feats_map.shape[:-1]
        input_label_shape = label_mask.shape
        
        # Downscale label mask
        zoom_factors = [x/y for x, y in zip(feats_map_shape, input_label_shape)]
        label_mask = zoom(label_mask, zoom=zoom_factors, order=0)
        
        # Choose segmentation method
        if mode == 'mlp':
            probs_map = _seg_via_mlp_head(label_mask, feats_map, num_epochs=num_epochs, return_prob=True)
        else:
            probs_map = _seg_via_conv_head(label_mask, feats_map, num_epochs=num_epochs, return_prob=True)
        
        # Upscale probability maps
        zoom_factors = [y/x for x, y in zip(feats_map_shape, input_label_shape)]
        zoomed_seg_prob = zoom(probs_map, zoom=(1, *zoom_factors), order=0)
        
        if return_prob:
            return zoomed_seg_prob
        else:
            pred_mask = np.argmax(zoomed_seg_prob, axis=0) + 1  # Convert to 1-based labels
            return pred_mask.astype(np.uint8)
    
    @staticmethod
    def grow_cut_segmentation(label_mask: np.ndarray, feats_map: np.ndarray,
                            max_iter: int = 1000, strength: float = 0.5) -> np.ndarray:
        """
        Grow-cut segmentation using cellular automata with N-dimensional features.
        
        Args:
            label_mask: Input label mask
            feats_map: N-dimensional feature map (H, W, C) where C is feature dimension
            max_iter: Maximum number of iterations
            strength: Strength parameter for grow-cut algorithm
            
        Returns:
            Segmentation result
        """
        feats_map_shape = feats_map.shape[:-1]
        input_label_shape = label_mask.shape
        
        # Downscale label mask to feature map resolution
        zoom_factors = [x/y for x, y in zip(feats_map_shape, input_label_shape)]
        label_mask = zoom(label_mask, zoom=zoom_factors, order=0)
        print(f"Downscaled label mask shape: {label_mask.shape}")
        print(f"Feature map shape: {feats_map.shape}")
        
        # Check for valid labels
        unique_labels = np.unique(label_mask)
        unique_labels = unique_labels[unique_labels != 0]  # Ignore background
        
        if len(unique_labels) < 2:
            print("Warning: Not enough labels for grow-cut")
            return np.zeros(label_mask.shape, dtype=np.uint8)
        
        try:
            # Run N-dimensional grow-cut algorithm
            print(f"Running N-dimensional grow-cut with max_iter={max_iter}, strength={strength}")
            result = SegmentationMethods._growcut_ndim(feats_map, label_mask, max_iter, strength)
            
            # Upscale result back to original resolution
            zoom_factors = [y/x for x, y in zip(feats_map_shape, input_label_shape)]
            zoomed_seg_label = zoom(result, zoom=zoom_factors, order=0)
            
            return zoomed_seg_label.astype(np.uint8)
            
        except Exception as e:
            print(f"Error in grow-cut segmentation: {e}")
            # Return original label mask if grow-cut fails
            return label_mask.astype(np.uint8)
    
    @staticmethod
    def _growcut_ndim(feats_map: np.ndarray, label_mask: np.ndarray, 
                     max_iter: int = 1000, strength: float = 0.5, window_size: int = 5) -> np.ndarray:
        """
        N-dimensional grow-cut segmentation using feature vector similarities.
        
        Args:
            feats_map: Feature map (H, W, C) where C is feature dimension
            label_mask: Label mask (H, W)
            max_iter: Maximum number of iterations
            strength: Strength parameter
            window_size: Neighborhood window size
            
        Returns:
            Segmentation result
        """
        height, width = feats_map.shape[:2]
        ws = (window_size - 1) // 2
        
        # Prepare state array: (H, W, 2) where [label, strength]
        state = np.zeros((height, width, 2), dtype=np.float32)
        
        # Set labels and strength for non-zero pixels
        unique_labels = np.unique(label_mask)
        unique_labels = unique_labels[unique_labels != 0]
        
        for label in unique_labels:
            mask = (label_mask == label)
            state[mask, 0] = label
            state[mask, 1] = strength
        
        changes = 1
        n = 0
        state_next = state.copy()
        
        while changes > 0 and n < max_iter:
            changes = 0
            n += 1
            
            if n % 10 == 0:
                print(f"Grow-cut iteration {n}")
            
            for j in range(width):
                for i in range(height):
                    # Current pixel feature vector
                    feat_p = feats_map[i, j]  # Shape: (C,)
                    state_p = state[i, j]
                    
                    # Check neighbors
                    for jj in range(max(0, j - ws), min(j + ws + 1, width)):
                        for ii in range(max(0, i - ws), min(i + ws + 1, height)):
                            # Neighbor feature vector
                            feat_q = feats_map[ii, jj]  # Shape: (C,)
                            state_q = state[ii, jj]
                            
                            # Compute feature similarity
                            similarity = SegmentationMethods._compute_feature_similarity(feat_q, feat_p)
                            
                            # Grow-cut condition: similarity * neighbor_strength > current_strength
                            if similarity * state_q[1] > state_p[1]:
                                state_next[i, j, 0] = state_q[0]  # Update label
                                state_next[i, j, 1] = similarity * state_q[1]  # Update strength
                                changes += 1
                                break
            
            state = state_next.copy()
        
        print(f"Grow-cut completed after {n} iterations")
        return state[:, :, 0]
    
    @staticmethod
    def _compute_feature_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute similarity between two feature vectors.
        
        Args:
            feat1: First feature vector (C,)
            feat2: Second feature vector (C,)
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize feature vectors
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
        
        # Compute cosine similarity
        cosine_sim = np.dot(feat1_norm, feat2_norm)
        
        # Convert to similarity score between 0 and 1
        # Higher cosine similarity -> higher similarity score
        similarity = (cosine_sim + 1) / 2  # Maps [-1, 1] to [0, 1]
        
        return similarity


class SegmentationInterface:
    """Main interface class that manages the GUI and segmentation workflow."""
    
    def __init__(self):
        # Initialize components
        self.model_manager = ModelManager()
        self.data_manager = DataManager(self.model_manager)
        self.seg_methods = SegmentationMethods()
        
        # State variables
        self.last_seg_data = None
        self.last_label_data = None
        self.current_label_data = None
        
        # Initialize GUI
        self._setup_gui()
        self._setup_event_handlers()
    
    def _setup_gui(self):
        """Setup the Napari viewer and control widgets."""
        # Create Napari viewer
        self.viewer = napari.Viewer(ndisplay=2)
        self.viewer.add_image(self.data_manager.z_slice, name='img')
        
        # Load initial label data
        label_data = self.data_manager.get_label_data()
        self.label_layer = self.viewer.add_labels(label_data, name='Label')
        self.label_layer.brush_size = 30
        self.label_layer.mode = 'PAINT'
        
        self.segout_layer = self.viewer.add_labels(label_data, name='Segout')
        self.viewer.layers.selection = [self.label_layer]
        
        # Initialize state variables
        self._reset_state_variables()
        
        # Create control widgets
        self._create_widgets()
        
        # Add control panel to viewer
        control_panel = Container(widgets=[
            self.dataset_button, self.model_button, self.method_button, self.pcafeats_button,
            self.epoch_choice, self.cut_lambda_slider, self.cut_sigma_slider,
            self.dinov3_layers, self.dinov3_use_pca, self.dinov3_pca_dim,
            self.growcut_max_iter, self.growcut_strength,
            self.seg_button, self.clear_button, self.undo_button
        ])
        
        self.viewer.window.add_dock_widget(control_panel, area='right')
        
        # Initialize widget visibility based on default model selection
        self._on_model_changed()
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        self.dataset_button = widgets.ComboBox(
            value='t1779', 
            choices=DatasetConfig.get_available_datasets(), 
            label='Dataset'
        )
        self.seg_button = widgets.PushButton(text="Seg")
        self.method_button = widgets.ComboBox(
            value='conv_seg', 
            choices=['graphcut', 'mlp_seg', 'conv_seg', 'similarity', 'growcut']
        )
        self.model_button = widgets.ComboBox(
            value='composite', 
            choices=['composite', 'dinov3'],
            label='Feature Model'
        )
        self.pcafeats_button = widgets.ComboBox(
            value='non-pca', 
            choices=['pca', 'non-pca', 'dinov3', 'dinov3-pca']
        )
        self.epoch_choice = widgets.ComboBox(
            label='epoch', 
            value=2000, 
            choices=[100, 1000, 2000, 5000, 10000]
        )
        self.cut_sigma_slider = widgets.ComboBox(
            label='sigma', 
            value=0.01, 
            choices=[0.001, 0.005, 0.01, 0.05]
        )
        self.cut_lambda_slider = widgets.ComboBox(
            label='lambda', 
            value=0.01, 
            choices=[0.01, 0.1, 0.5, 1, 1.5, 2, 10]
        )
        self.dinov3_layers = widgets.ComboBox(
            label='DINOv3 Layers', 
            value='2,5,8,11', 
            choices=['5','7', '11','12', '2,11','2,5,11','2,5,8,11','1,3,6,9',]
        )
        self.dinov3_use_pca = widgets.CheckBox(
            label='Use DINOv3 PCA', 
            value=True
        )
        self.dinov3_pca_dim = widgets.ComboBox(
            label='DINOv3 PCA Dim', 
            value=256, 
            choices=[64, 128, 256, 512, 1024]
        )
        self.growcut_max_iter = widgets.ComboBox(
            label='Grow-cut Max Iter', 
            value=1000, 
            choices=[100, 500, 1000, 2000, 5000]
        )
        self.growcut_strength = widgets.ComboBox(
            label='Grow-cut Strength', 
            value=0.5, 
            choices=[0.1, 0.3, 0.5, 0.7, 0.9]
        )
        self.clear_button = widgets.PushButton(text="Clear")
        self.undo_button = widgets.PushButton(text="Undo")
    
    def _setup_event_handlers(self):
        """Setup event handlers for all widgets."""
        self.dataset_button.changed.connect(self._on_dataset_changed)
        self.model_button.changed.connect(self._on_model_changed)
        self.method_button.changed.connect(self._on_method_changed)
        self.seg_button.clicked.connect(self._on_seg_button_clicked)
        self.clear_button.clicked.connect(self._on_clear_button_clicked)
        self.undo_button.clicked.connect(self._on_undo_button_clicked)
        self.dinov3_layers.changed.connect(self._on_dinov3_params_changed)
        self.dinov3_use_pca.changed.connect(self._on_dinov3_pca_toggled)
        self.dinov3_pca_dim.changed.connect(self._on_dinov3_params_changed)
    
    def _reset_state_variables(self):
        """Reset state variables for current dataset."""
        shape = self.data_manager.input_label_shape
        self.last_seg_data = np.zeros(shape, dtype=np.uint8)
        self.last_label_data = np.zeros(shape, dtype=np.uint8)
        self.current_label_data = np.zeros(shape, dtype=np.uint8)
    
    def _on_dataset_changed(self):
        """Handle dataset change event."""
        new_dataset = self.dataset_button.value
        if new_dataset != self.data_manager.current_dataset:
            print(f"Switching to dataset: {new_dataset}")
            
            # Load new dataset
            self.data_manager.load_dataset(new_dataset)
            
            # Update viewer
            self.viewer.layers['img'].data = self.data_manager.z_slice
            
            # Load new label data
            label_data = self.data_manager.get_label_data()
            self.label_layer.data = label_data
            self.segout_layer.data = label_data
            
            # Reset state
            self._reset_state_variables()
            
            print(f"Dataset switched to: {new_dataset}")
    
    def _on_model_changed(self):
        """Handle model selection changes."""
        selected_model = self.model_button.value
        print(f"Model changed to: {selected_model}")
        
        # Update feature options based on model selection
        if selected_model == 'composite':
            # Show composite model options
            self.pcafeats_button.choices = ['pca', 'non-pca']
            self.pcafeats_button.value = 'non-pca'
            # Hide DINOv3 specific options
            self.dinov3_layers.visible = False
            self.dinov3_use_pca.visible = False
            self.dinov3_pca_dim.visible = False
        else:  # dinov3
            # Show DINOv3 model options
            self.pcafeats_button.choices = ['dinov3', 'dinov3-pca']
            self.pcafeats_button.value = 'dinov3'
            # Show DINOv3 specific options
            self.dinov3_layers.visible = True
            self.dinov3_use_pca.visible = True
            self.dinov3_pca_dim.visible = True
        
        # Also update method-specific widget visibility
        self._on_method_changed()
    
    def _on_method_changed(self):
        """Handle segmentation method changes."""
        selected_method = self.method_button.value
        print(f"Method changed to: {selected_method}")
        
        # Show/hide method-specific widgets
        if selected_method == 'growcut':
            self.growcut_max_iter.visible = True
            self.growcut_strength.visible = True
        else:
            self.growcut_max_iter.visible = False
            self.growcut_strength.visible = False
    
    def _on_dinov3_pca_toggled(self):
        """Handle DINOv3 PCA toggle changes."""
        # Enable/disable PCA dimension widget based on PCA checkbox
        self.dinov3_pca_dim.enabled = self.dinov3_use_pca.value
        
        # Also recompute features with new PCA setting
        self._on_dinov3_params_changed()
    
    def _on_seg_button_clicked(self):
        """Handle segmentation button click."""
        label_data = self.label_layer.data.copy()
        
        # Update state variables
        self.last_label_data = self.current_label_data
        self.current_label_data = label_data
        self.last_seg_data = self.segout_layer.data.copy()
        
        # Get parameters
        mode = self.method_button.value
        selected_model = self.model_button.value
        feats_map_mode = self.pcafeats_button.value
        num_epoch = self.epoch_choice.value
        
        # Select feature map based on model choice
        if selected_model == 'composite':
            # Use composite model features
            if feats_map_mode == 'pca':
                feats = self.data_manager.rgb_vis
            else:  # non-pca
                feats = self.data_manager.feats_map
        else:  # dinov3
            # Use DINOv3 model features
            if feats_map_mode == 'dinov3-pca':
                feats = self.data_manager.dinov3_rgb_vis
            else:  # dinov3
                feats = self.data_manager.dinov3_feats
        
        # Perform segmentation
        start_time = time.time()
        print(f"Starting segmentation with method: {mode}")
        
        try:
            if mode == "graphcut":
                seg_result = self.seg_methods.graph_cut_segmentation(
                    label_data, feats, 
                    self.cut_sigma_slider.value, 
                    self.cut_lambda_slider.value
                )
            elif mode == 'mlp_seg':
                seg_result = self.seg_methods.head_based_segmentation(
                    label_data, feats, num_epochs=num_epoch, mode='mlp'
                )
            elif mode == 'conv_seg':
                seg_result = self.seg_methods.head_based_segmentation(
                    label_data, feats, num_epochs=num_epoch, mode='conv'
                )
            elif mode == 'similarity':
                seg_result = self.seg_methods.similarity_segmentation(
                    label_data, feats, spatial_decay=True
                )
            elif mode == 'growcut':
                seg_result = self.seg_methods.grow_cut_segmentation(
                    label_data, feats,
                    max_iter=self.growcut_max_iter.value,
                    strength=self.growcut_strength.value
                )
            else:
                print(f"Unknown segmentation method: {mode}")
                return
            
            # Update segmentation result
            self.segout_layer.data = seg_result
            elapsed_time = time.time() - start_time
            print(f"Segmentation completed in {elapsed_time:.3f} seconds")
            
        except Exception as e:
            print(f"Error during segmentation: {e}")
            import traceback
            traceback.print_exc()
        
        # Keep label layer selected
        self.viewer.layers.selection = [self.label_layer]
    
    def _on_clear_button_clicked(self):
        """Handle clear button click."""
        self.label_layer.data = np.zeros_like(self.label_layer.data)
        self.segout_layer.data = np.zeros_like(self.segout_layer.data)
        self.viewer.layers.selection = [self.label_layer]
    
    def _on_undo_button_clicked(self):
        """Handle undo button click."""
        self.label_layer.data = self.last_label_data
        self.segout_layer.data = self.last_seg_data
        self.viewer.layers.selection = [self.label_layer]
    
    def _on_dinov3_params_changed(self):
        """Handle DINOv3 parameter changes."""
        print("DINOv3 parameters changed, recomputing features...")
        
        # Parse layer indices
        layer_str = self.dinov3_layers.value
        layer_indices = [int(x.strip()) for x in layer_str.split(',')]
        
        # Get PCA settings
        use_pca = self.dinov3_use_pca.value
        pca_dim = self.dinov3_pca_dim.value if use_pca else None
        
        # Recompute DINOv3 features using the data manager method
        self.data_manager.recompute_dinov3_features(layer_indices=layer_indices, pca_dim=pca_dim)
    
    def run(self):
        """Start the application."""
        napari.run()


# Main execution
if __name__ == "__main__":
    # Create and run the interface
    interface = SegmentationInterface()
    interface.run()