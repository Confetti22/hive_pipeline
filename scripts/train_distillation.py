"""
# Old way (distillation.py):
# Edit the file to change settings, then:
python distillation.py

# New way (scripts/train_distillation.py):
# Edit config/distill.yaml, then:
python scripts/train_distillation.py -cfg config/distill.yaml

"""

import argparse
import os
import re
import sys
import yaml
from glob import glob
from itertools import chain
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

# Get the path to the parent directory of 'scripts', which is 'hive1_pipeline'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from lib.distill import GrayTiffDataset, Distiller
from lib.distill.student import build_student_cnn
from lib.utils.augmentations import GPUAugmentations
from lib.core.optimizer import get_parameter_groups

def _validate_paths(paths: List[str], max_count: int | None = None) -> List[str]:
    files: List[str] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            # collect tif/tiff files in the directory
            files.extend([str(f) for f in pp.rglob('*.tif')])
            files.extend([str(f) for f in pp.rglob('*.tiff')])
        elif pp.is_file():
            files.append(str(pp))
    files = sorted(files)
    if max_count is not None:
        files = files[:max_count]
        print(f"[DATA] Limiting to first {len(files)} files for this run")
    return files


def load_latest_epoch(distiller: Distiller, save_dir: str):
    """
    Loads the latest student checkpoint. 
    Returns 0 if no checkpoints exist, allowing for a fresh training start.
    """
    save_path = Path(save_dir)
    
    # Ensure the directory exists; if not, there are definitely no files
    if not save_path.exists():
        print(f"[INFO] Directory {save_dir} not found. Starting from epoch 0.")
        return 0

    # Find all student_epoch_XXX.pth files
    checkpoints = list(save_path.glob("student_epoch_*.pth"))
    
    if not checkpoints:
        print(f"[INFO] No existing checkpoints found in {save_dir}. Starting from epoch 0.")
        return 0

    try:
        # Sort by the numeric value in the filename to avoid "10" coming before "2"
        checkpoints.sort(key=lambda x: int(re.findall(r'\d+', x.name)[-1]))
        latest_checkpoint = checkpoints[-1]
        
        # Load the weights
        state_dict = torch.load(latest_checkpoint, map_location='cpu', weights_only=True)
        distiller.student.load_state_dict(state_dict)
        
        # Extract epoch number from filename (e.g., '015' -> 15)
        last_epoch = int(re.findall(r'\d+', latest_checkpoint.name)[-1])
        
        print(f"[LOAD] Resumed from {latest_checkpoint.name} (Epoch {last_epoch})")
        return last_epoch

    except Exception as e:
        print(f"[WARNING] Error loading checkpoint: {e}. Falling back to epoch 0.")
        return 0



def main():
    p = argparse.ArgumentParser()
    p.add_argument('-cfg', default='config/distill.yaml', help='Path to distillation YAML config')
    args = p.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    raw_paths = cfg.get('train_paths', [])
    max_train_samples = cfg.get('max_train_samples', None)
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
    train_paths = _validate_paths(raw_paths, max_train_samples)
    if not train_paths:
        raise SystemExit("[ERR] No training images found. Provide train_paths (files or directories) in the config.")
    teacher_dir = cfg['teacher_dir']
    ckpt_path = cfg.get('ckpt_path', None)
    student_type = cfg.get('student_type', 'cnn')
    tinyvit_input_type = cfg.get('tinyvit_input_type', 'rgb')
    tinyvit_implementation = cfg.get('tinyvit_implementation', 'local')
    tinyvit_depth = int(cfg.get('tinyvit_depth', 12))
    tinyvit_embed_dim = int(cfg.get('tinyvit_embed_dim', 192))
    tinyvit_num_heads = cfg.get('tinyvit_num_heads', None)
    if tinyvit_num_heads is not None:
        tinyvit_num_heads = int(tinyvit_num_heads)
    tinyvit_mlp_ratio = float(cfg.get('tinyvit_mlp_ratio', 4.0))
    teacher_feature_layers = cfg.get('teacher_feature_layers', [2, 6, 11])
    student_feature_layers = cfg.get('student_feature_layers', [2, 6, 11])
    lambda_feat = float(cfg.get('lambda_feat', 1.0))
    lambda_aff = float(cfg.get('lambda_aff', 0.25))
    use_spatial_coords = bool(cfg.get('use_spatial_coords', False))
    feature_loss_type = cfg.get('feature_loss_type', 'cosine')  # 'cosine' or 'mse'
    crop_size = cfg.get('crop_size', None)
    if crop_size is not None:
        crop_size = int(crop_size)
    batch_size = int(cfg.get('batch_size', 8))
    epochs = int(cfg.get('epochs', 50))
    lr = float(cfg.get('lr', 5e-4))
    wd = float(cfg.get('weight_decay', 0.05))
    num_workers = int(cfg.get('num_workers', 4))
    
    # Learning rate scheduler configuration
    scheduler_type = cfg.get('scheduler_type', 'cosine')
    scheduler_warmup_epochs = int(cfg.get('scheduler_warmup_epochs', 5))
    scheduler_min_lr = float(cfg.get('scheduler_min_lr', 1e-6))
    use_mixup = bool(cfg.get('use_mixup', False))
    use_aug = bool(cfg.get('use_aug', False))
    
    # Model saving configuration
    save_every_epoch = int(cfg.get('save_every_epoch', 0))
    save_dir = f"./runs/distill/{cfg['student_type']}_aug_{cfg['use_aug']}_no_norm_wd_rm009v1_t1779"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create save directory if saving is enabled
    if save_every_epoch > 0:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(f"[SAVE] Model saving enabled: every {save_every_epoch} epochs to {save_dir}")
    else:
        print(f"[SAVE] Model saving disabled")
    
    # Print PyTorch and CUDA version info for debugging
    print(f"[SYSTEM] PyTorch version: {torch.__version__}")
    print(f"[SYSTEM] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[SYSTEM] CUDA version: {torch.version.cuda}")
        print(f"[SYSTEM] cuDNN version: {torch.backends.cudnn.version()}")
        print(f"[SYSTEM] CUDA device capability: {torch.cuda.get_device_capability()}")

    ds = GrayTiffDataset(train_paths)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    augmentor = GPUAugmentations(size=crop_size, v=False).to(device)

    distiller = Distiller(
        teacher_dir=teacher_dir,
        ckpt_path=ckpt_path,
        student_type=student_type,
        student_cnn_builder=build_student_cnn(model_type='simple') if student_type == 'cnn' else None,
        tinyvit_input_type=tinyvit_input_type,
        tinyvit_implementation=tinyvit_implementation,
        tinyvit_depth=tinyvit_depth,
        tinyvit_embed_dim=tinyvit_embed_dim,
        tinyvit_num_heads=tinyvit_num_heads,
        tinyvit_mlp_ratio=tinyvit_mlp_ratio,
        teacher_feature_layers=teacher_feature_layers,
        student_feature_layers=student_feature_layers,
        lambda_feat=lambda_feat,
        lambda_aff=lambda_aff,
        use_spatial_coords=use_spatial_coords,
        feature_loss_type=feature_loss_type,
    ).to(device)

    from lib.distill.teacher import count_model_size
    count_model_size(distiller.teacher)
    count_model_size(distiller.student)

    print(distiller.student)
    writer = SummaryWriter(log_dir=save_dir)

    # Log configuration
    print(f"[CONFIG] Student type: {student_type}")
    if student_type == "tinyvit":
        print(f"[CONFIG] TinyViT implementation: {tinyvit_implementation}")
        print(f"[CONFIG] TinyViT input type: {tinyvit_input_type}")
        print(f"[CONFIG] TinyViT model type: {distiller.student.input_type}")
        print(f"[CONFIG] TinyViT depth: {tinyvit_depth}")
        print(f"[CONFIG] TinyViT embed_dim: {tinyvit_embed_dim}")
        print(f"[CONFIG] TinyViT num_heads: {tinyvit_num_heads}")
        print(f"[CONFIG] TinyViT mlp_ratio: {tinyvit_mlp_ratio}")
    print(f"[CONFIG] Teacher feature layers: {teacher_feature_layers}")
    print(f"[CONFIG] Student feature layers: {student_feature_layers}")
    print(f"[CONFIG] Loss weights - Feature: {lambda_feat}, Affinity: {lambda_aff}")
    print(f"[CONFIG] Feature loss type: {feature_loss_type}")
    print(f"[CONFIG] Use spatial coords: {use_spatial_coords}")
    print(f"[CONFIG] Crop size: {crop_size if crop_size is not None else 'None (no cropping)'}")
    print(f"[CONFIG] Mixup: {'enabled' if use_mixup else 'disabled'}")
    print(f"[CONFIG] Batch size: {batch_size}, Epochs: {epochs}, LR: {lr}")
    print(f"[CONFIG] Scheduler: {scheduler_type}, Warmup epochs: {scheduler_warmup_epochs}, Min LR: {scheduler_min_lr}")
    
    # Device status
    is_cuda = device.startswith('cuda')
    print(f"[CONFIG] Device: {device}")
    if is_cuda:
        print(f"[CONFIG] CUDA Device: {torch.cuda.get_device_name()}")
        print(f"[CONFIG] CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Separate parameters into weight decay and no weight decay groups
    optim_groups = get_parameter_groups(distiller.student, distiller.adapter, wd)
    print(f"Separate parameters into weight decay and no weight decay groups")
    
    optim = torch.optim.AdamW(
        optim_groups,
        lr=lr,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=False)  # Disabled mixed precision training
    
    # Create learning rate scheduler
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=epochs, eta_min=scheduler_min_lr
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=epochs//3, gamma=0.1
        )
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optim, gamma=0.95
        )
    elif scheduler_type == 'cosine_warmup':
        # Custom cosine with warmup
        def lr_lambda(epoch):
            if epoch < scheduler_warmup_epochs:
                return epoch / scheduler_warmup_epochs
            else:
                progress = (epoch - scheduler_warmup_epochs) / (epochs - scheduler_warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    else:
        scheduler = None

    distiller.teacher.eval()
    distiller.student.train()

    st_epoch = load_latest_epoch(distiller, save_dir)
    end_epoch = epochs

    for ep in range(st_epoch, end_epoch):
        for batch_idx, (x_rgb, image_ids) in enumerate(dl):
            x_rgb = x_rgb.to(device, non_blocking=True)
            # data augmentation
            if use_aug:
                x_rgb = augmentor(x_rgb)

            if use_mixup:
                # Mixup coefficients sampled uniformly in [0, 1]
                lam = torch.rand(x_rgb.size(0), device=device, dtype=x_rgb.dtype).view(-1, 1, 1, 1)
                perm = torch.randperm(x_rgb.size(0), device=device)
                x_rgb = lam * x_rgb + (1 - lam) * x_rgb[perm]

            optim.zero_grad(set_to_none=True)

            # Debug: Check input dtype before distiller
            if ep == 0 and batch_idx == 0:
                print(f"[DEBUG] - Input to distiller dtype: {x_rgb.dtype}")
            
            out = distiller(x_rgb, image_ids)
            loss = out['loss']
            
            # Debug: Check intermediate results
            if ep == 0 and batch_idx == 0:
                print(f"[DEBUG] - Distiller output keys: {list(out.keys())}")
                for key, value in out.items():
                    if isinstance(value, torch.Tensor):
                        print(f"[DEBUG] - {key} dtype: {value.dtype}")
                
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        
        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step()
        
        current_lr = optim.param_groups[0]['lr']
        print(f"[distill] epoch={ep+1:03d} loss={loss.item():.4f} L_feat={out['L_feat'].item():.4f} L_aff={out['L_aff'].item():.4f} lr={current_lr:.6f}")

        writer.add_scalar("train/distill_loss", out['L_feat'].item(), ep)
        
        # Save student weights at specified intervals
        if save_every_epoch > 0 and (ep + 1) % save_every_epoch == 0:
            save_path = Path(save_dir) / f"student_epoch_{ep+1:03d}.pth"
            torch.save(distiller.student.state_dict(), save_path)
            print(f"[SAVE] Saved student weights to {save_path}")
    
    # Save final model
    if save_every_epoch > 0:
        final_save_path = Path(save_dir) / "student_final.pth"
        torch.save(distiller.student.state_dict(), final_save_path)
        print(f"[SAVE] Saved final student weights to {final_save_path}")


if __name__ == '__main__':
    main()
