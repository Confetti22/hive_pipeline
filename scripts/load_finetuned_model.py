"""
Utility script to load and test a finetuned DINOv3 model.

Usage:
    python scripts/load_finetuned_model.py -model path/to/checkpoint.pth -image path/to/image.tif
"""

import sys
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import tifffile as tiff
from transformers import AutoModel

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from lib.distill.data import preprocess_uint16_for_imagenet, to_rgb_for_vit


class DINOv3FinetuneModel(nn.Module):
    """DINOv3 model wrapper for finetuning on single-channel images."""
    
    def __init__(self, model_dir: str, freeze_backbone: bool = False, freeze_layers: int = 0):
        super().__init__()
        
        # Load pretrained DINOv3 model
        self.vit = AutoModel.from_pretrained(
            model_dir, 
            local_files_only=True, 
            output_hidden_states=True
        )
        
        # Configure freezing
        if freeze_backbone:
            # Freeze all parameters except the last layer
            for param in self.vit.parameters():
                param.requires_grad = False
            # Unfreeze the last layer
            for param in self.vit.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif freeze_layers > 0:
            # Freeze specified number of layers from the beginning
            for i in range(freeze_layers):
                for param in self.vit.encoder.layer[i].parameters():
                    param.requires_grad = False
        
        self.embed_dim = self.vit.config.hidden_size
        
        # Add a projection head for self-supervised learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 256),  # Project to 256-dim representation
        )
        
    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass through the model."""
        # Get hidden states from the model
        outputs = self.vit(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Use the last hidden state
        last_hidden = hidden_states[-1]  # [B, N_patches+1, embed_dim]
        
        # Extract CLS token (first token)
        cls_token = last_hidden[:, 0]  # [B, embed_dim]
        
        # Project features
        projections = self.projection_head(cls_token)  # [B, 256]
        
        return {
            'features': last_hidden,
            'projections': projections,
            'cls_token': cls_token
        }


def load_checkpoint(checkpoint_path: str, model_dir: str, device: str = 'cpu'):
    """Load a finetuned model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create model
    model = DINOv3FinetuneModel(
        model_dir=model_dir,
        freeze_backbone=config.get('freeze_backbone', False),
        freeze_layers=config.get('freeze_layers', 0)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def preprocess_image(image_path: str, crop_size: int = None) -> torch.Tensor:
    """Preprocess a single TIFF image for inference."""
    # Load image
    img = tiff.imread(image_path)
    
    if len(img.shape) == 3:
        img = img[0]  # Take first channel if multi-channel
    
    # Apply central crop if specified
    if crop_size is not None:
        h, w = img.shape
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        end_h = start_h + crop_size
        end_w = start_w + crop_size
        
        # Ensure we don't go out of bounds
        start_h = max(0, start_h)
        start_w = max(0, start_w)
        end_h = min(h, end_h)
        end_w = min(w, end_w)
        
        img = img[start_h:end_h, start_w:end_w]
        
        # Pad if necessary
        if img.shape[0] < crop_size or img.shape[1] < crop_size:
            padded = np.zeros((crop_size, crop_size), dtype=img.dtype)
            pad_h = (crop_size - img.shape[0]) // 2
            pad_w = (crop_size - img.shape[1]) // 2
            padded[pad_h:pad_h+img.shape[0], pad_w:pad_w+img.shape[1]] = img
            img = padded
    
    # Preprocess for ImageNet normalization
    x = preprocess_uint16_for_imagenet(img)  # [C, D, H, W]
    x = x.squeeze(1)  # [C, H, W]
    
    # Convert to RGB
    x_rgb = to_rgb_for_vit(x.unsqueeze(0))  # [1, 3, H, W]
    
    return x_rgb


def main():
    parser = argparse.ArgumentParser(description='Load and test a finetuned DINOv3 model')
    parser.add_argument('-model', required=True, help='Path to finetuned model checkpoint')
    parser.add_argument('-image', required=True, help='Path to test image')
    parser.add_argument('-model_dir', required=True, help='Path to original DINOv3 model directory')
    parser.add_argument('-device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('-crop_size', type=int, default=None, help='Crop size for preprocessing')
    
    args = parser.parse_args()
    
    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"[INFO] Using device: {device}")
    
    # Load model
    print(f"[INFO] Loading model from {args.model}")
    model, config = load_checkpoint(args.model, args.model_dir, device)
    print(f"[INFO] Model loaded successfully")
    print(f"[INFO] Model config: {config}")
    
    # Preprocess image
    print(f"[INFO] Loading and preprocessing image: {args.image}")
    x_rgb = preprocess_image(args.image, args.crop_size)
    x_rgb = x_rgb.to(device)
    
    print(f"[INFO] Input shape: {x_rgb.shape}")
    
    # Run inference
    print(f"[INFO] Running inference...")
    with torch.no_grad():
        outputs = model(x_rgb)
    
    # Print results
    print(f"\n[RESULTS]")
    print(f"CLS token shape: {outputs['cls_token'].shape}")
    print(f"Projections shape: {outputs['projections'].shape}")
    print(f"Features shape: {outputs['features'].shape}")
    
    # Print some statistics
    cls_token = outputs['cls_token'].cpu().numpy()
    projections = outputs['projections'].cpu().numpy()
    
    print(f"\n[STATISTICS]")
    print(f"CLS token - Mean: {cls_token.mean():.4f}, Std: {cls_token.std():.4f}")
    print(f"CLS token - Min: {cls_token.min():.4f}, Max: {cls_token.max():.4f}")
    print(f"Projections - Mean: {projections.mean():.4f}, Std: {projections.std():.4f}")
    print(f"Projections - Min: {projections.min():.4f}, Max: {projections.max():.4f}")
    
    # Save features if requested
    save_features = input("\nSave features to file? (y/n): ").lower().strip() == 'y'
    if save_features:
        output_path = Path(args.image).stem + '_features.npz'
        np.savez(output_path, 
                cls_token=cls_token,
                projections=projections,
                features=outputs['features'].cpu().numpy())
        print(f"[INFO] Features saved to {output_path}")


if __name__ == '__main__':
    main()
