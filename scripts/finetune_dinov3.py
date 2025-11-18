"""
DINOv3 Finetuning Script for Single-Channel uint16 GrayTiffDataset

Usage:
    python scripts/finetune_dinov3.py -cfg config/finetune.yaml

This script finetunes a pretrained DINOv3 model on single-channel uint16 TIFF images.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig
import numpy as np
from tqdm import tqdm

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from lib.distill.data import GrayTiffDataset, to_rgb_for_vit


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
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [B, 3, H, W] (RGB images)
            
        Returns:
            Dictionary containing:
                - 'features': Final hidden states [B, N_patches+1, embed_dim]
                - 'projections': Projected features [B, 256]
                - 'cls_token': CLS token [B, embed_dim]
        """
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


class DINOLoss(nn.Module):
    """DINO-style self-supervised loss for finetuning."""
    
    def __init__(self, temperature: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.temperature = temperature
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 256))
        
    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """
        Compute DINO loss between student and teacher outputs.
        
        Args:
            student_output: Student projections [B, 256]
            teacher_output: Teacher projections [B, 256]
            
        Returns:
            DINO loss scalar
        """
        # Normalize outputs
        student_out = F.normalize(student_output, dim=-1)
        teacher_out = F.normalize(teacher_output, dim=-1)
        
        # Update center - ensure center is on the same device as teacher_out
        with torch.no_grad():
            batch_center = teacher_out.mean(dim=0, keepdim=True)
            self.center = self.center_momentum * self.center + (1 - self.center_momentum) * batch_center
        
        # Compute loss
        student_centered = student_out - self.center
        teacher_centered = teacher_out - self.center
        
        # Cross-entropy loss
        logits = torch.matmul(student_centered, teacher_centered.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        return loss / 2


def validate_paths(paths: List[str]) -> List[str]:
    """Validate and collect TIFF file paths."""
    files: List[str] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            # Collect tif/tiff files in the directory
            files.extend([str(f) for f in pp.rglob('*.tif')])
            files.extend([str(f) for f in pp.rglob('*.tiff')])
        elif pp.is_file():
            files.append(str(pp))
    return sorted(files)


def create_optimizer_and_scheduler(model: nn.Module, config: Dict[str, Any]) -> tuple:
    """Create optimizer and learning rate scheduler."""
    # Ensure numeric values are properly converted
    learning_rate = float(config['learning_rate'])
    weight_decay = float(config['weight_decay'])
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = None
    if config['scheduler_type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config['epochs']), eta_min=float(config['scheduler_min_lr'])
        )
    elif config['scheduler_type'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(config['epochs'])//3, gamma=0.1
        )
    elif config['scheduler_type'] == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95
        )
    elif config['scheduler_type'] == 'cosine_warmup':
        warmup_epochs = int(config['scheduler_warmup_epochs'])
        total_epochs = int(config['epochs'])
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, device: torch.device, config: Dict[str, Any]) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (x_gray, image_ids) in enumerate(pbar):
        x_gray = x_gray.to(device, non_blocking=True)
        
        # Debug: Print tensor shapes for first batch
        if batch_idx == 0:
            print(f"[DEBUG] x_gray shape: {x_gray.shape}")
            print(f"[DEBUG] x_gray dtype: {x_gray.dtype}")
        
        # Convert grayscale to RGB for DINOv3
        x_rgb = to_rgb_for_vit(x_gray)
        
        # Debug: Print RGB tensor shape for first batch
        if batch_idx == 0:
            print(f"[DEBUG] x_rgb shape: {x_rgb.shape}")
            print(f"[DEBUG] x_rgb dtype: {x_rgb.dtype}")
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x_rgb)
        projections = outputs['projections']
        
        # For self-supervised learning, we use the same projections as both student and teacher
        # In a more sophisticated setup, you might use different augmentations or momentum updates
        loss = criterion(projections, projections.detach())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        log_every_n_steps = int(config.get('log_every_n_steps', 50))
        if batch_idx % log_every_n_steps == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {
        'loss': total_loss / num_batches,
        'num_batches': num_batches
    }


def validate_epoch(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for x_gray, image_ids in pbar:
            x_gray = x_gray.to(device, non_blocking=True)
            x_rgb = to_rgb_for_vit(x_gray)
            
            outputs = model(x_rgb)
            projections = outputs['projections']
            
            # Simple validation loss (L2 norm of projections)
            loss = torch.mean(torch.norm(projections, dim=1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'num_batches': num_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Finetune DINOv3 on single-channel uint16 TIFF images')
    parser.add_argument('-cfg', default='config/finetune.yaml', help='Path to finetune YAML config')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    # Validate training paths
    raw_paths = config.get('train_paths', [])
    train_paths = validate_paths(raw_paths)
    if not train_paths:
        raise SystemExit("[ERR] No training images found. Provide train_paths in the config.")
    
    # Validate validation paths if provided
    val_paths = []
    if config.get('val_paths'):
        val_paths = validate_paths(config['val_paths'])
        if not val_paths:
            print("[WARN] No validation images found. Skipping validation.")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[SYSTEM] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[SYSTEM] CUDA device: {torch.cuda.get_device_name()}")
        print(f"[SYSTEM] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create datasets
    crop_size = config.get('crop_size', None)
    if crop_size is not None and crop_size != 'null':
        crop_size = int(crop_size)
    else:
        crop_size = None
    
    train_dataset = GrayTiffDataset(train_paths, crop_size=crop_size)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=int(config['batch_size']), 
        shuffle=True, 
        num_workers=int(config['num_workers']), 
        pin_memory=True
    )
    
    val_dataloader = None
    if val_paths:
        val_dataset = GrayTiffDataset(val_paths, crop_size=crop_size)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=int(config['batch_size']),
            shuffle=False,
            num_workers=int(config['num_workers']),
            pin_memory=True
        )
    
    # Create model
    model = DINOv3FinetuneModel(
        model_dir=config['model_dir'],
        freeze_backbone=bool(config.get('freeze_backbone', False)),
        freeze_layers=int(config.get('freeze_layers', 0))
    ).to(device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # Create loss function
    criterion = DINOLoss().to(device)
    
    # Create save directory
    save_dir = config.get('save_dir', './finetune_checkpoints')
    save_every_epoch = int(config.get('save_every_epoch', 0))
    if save_every_epoch > 0:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(f"[SAVE] Model saving enabled: every {save_every_epoch} epochs to {save_dir}")
    else:
        print(f"[SAVE] Model saving disabled")
    
    # Print configuration
    print(f"[CONFIG] Model directory: {config['model_dir']}")
    print(f"[CONFIG] Training images: {len(train_paths)}")
    print(f"[CONFIG] Validation images: {len(val_paths) if val_paths else 0}")
    print(f"[CONFIG] Crop size: {crop_size if crop_size is not None else 'None'}")
    print(f"[CONFIG] Batch size: {config['batch_size']}, Epochs: {config['epochs']}")
    print(f"[CONFIG] Learning rate: {config['learning_rate']}")
    print(f"[CONFIG] Freeze backbone: {config.get('freeze_backbone', False)}")
    print(f"[CONFIG] Freeze layers: {config.get('freeze_layers', 0)}")
    
    # Training loop
    epochs = int(config['epochs'])
    print(f"\n[INFO] Starting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        train_metrics = train_epoch(model, train_dataloader, optimizer, criterion, device, config)
        
        # Validation
        val_metrics = {}
        val_every_epoch = int(config.get('val_every_epoch', 5))
        if val_dataloader is not None and (epoch + 1) % val_every_epoch == 0:
            val_metrics = validate_epoch(model, val_dataloader, device)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        # Print metrics
        print(f"[EPOCH {epoch+1:03d}] "
              f"train_loss={train_metrics['loss']:.4f} "
              f"lr={current_lr:.6f} "
              f"time={epoch_time:.1f}s")
        
        if val_metrics:
            print(f"[EPOCH {epoch+1:03d}] "
                  f"val_loss={val_metrics['loss']:.4f}")
        
        # Save model
        if save_every_epoch > 0 and (epoch + 1) % save_every_epoch == 0:
            save_path = Path(save_dir) / f"dinov3_epoch_{epoch+1:03d}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics.get('loss', None),
                'config': config
            }, save_path)
            print(f"[SAVE] Saved checkpoint to {save_path}")
    
    # Save final model
    if save_every_epoch > 0:
        final_save_path = Path(save_dir) / "dinov3_final.pth"
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config
        }, final_save_path)
        print(f"[SAVE] Saved final model to {final_save_path}")
    
    total_time = time.time() - start_time
    print(f"\n[INFO] Training completed in {total_time/3600:.2f} hours")


if __name__ == '__main__':
    main()
