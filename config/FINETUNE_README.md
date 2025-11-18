# DINOv3 Finetuning Module

This module provides functionality to finetune DINOv3 models on single-channel uint16 GrayTiffDataset images.

## Files

- `config/finetune.yaml` - Configuration file for finetuning
- `scripts/finetune_dinov3.py` - Main finetuning script
- `scripts/load_finetuned_model.py` - Utility to load and test finetuned models

## Quick Start

### 1. Configure Training

Edit `config/finetune.yaml` to set your training parameters:

```yaml
# Required: Path to the pretrained DINOv3 model
model_dir: /path/to/your/dinov3-vits16-pretrain-lvd1689m

# Training data paths
train_paths: [
    "/path/to/your/tiff/images",
    # Add more paths as needed
]

# Training configuration
batch_size: 32
epochs: 100
learning_rate: 1e-4
crop_size: 224  # Optional central crop
```

### 2. Run Finetuning

```bash
python scripts/finetune_dinov3.py -cfg config/finetune.yaml
```

### 3. Test Finetuned Model

```bash
python scripts/load_finetuned_model.py -model ./finetune_checkpoints/dinov3_final.pth -image /path/to/test_image.tif -model_dir /path/to/original/dinov3
```

## Configuration Options

### Data Configuration
- `train_paths`: List of paths to training TIFF images (files or directories)
- `val_paths`: List of paths to validation TIFF images (optional)
- `crop_size`: Optional central crop size (224, 256, 512, etc.)

### Model Configuration
- `model_dir`: Path to pretrained DINOv3 model directory
- `freeze_backbone`: If true, only finetune the final layers
- `freeze_layers`: Number of layers to freeze from the beginning (0 = no freezing)

### Training Configuration
- `batch_size`: Training batch size
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimization
- `weight_decay`: Weight decay for regularization
- `num_workers`: Number of data loading workers

### Learning Rate Scheduler
- `scheduler_type`: Type of scheduler (cosine, step, exponential, cosine_warmup, none)
- `scheduler_warmup_epochs`: Number of warmup epochs (for cosine_warmup)
- `scheduler_min_lr`: Minimum learning rate

### Model Saving
- `save_every_epoch`: Save model every N epochs (0 to disable)
- `save_dir`: Directory to save model checkpoints

### Validation
- `val_paths`: Paths to validation data (empty list to skip)
- `val_every_epoch`: Run validation every N epochs (0 to disable)

## Model Architecture

The finetune module uses a modified DINOv3 architecture:

1. **Base DINOv3**: Loads pretrained DINOv3-vits16 model
2. **Projection Head**: Adds a 2-layer MLP projection head (embed_dim → embed_dim → 256)
3. **Self-Supervised Loss**: Uses DINO-style contrastive loss for self-supervised learning

## Key Features

### Data Handling
- **Single-channel support**: Automatically converts grayscale TIFF to RGB for DINOv3
- **uint16 support**: Handles uint16 TIFF images with proper normalization
- **Flexible cropping**: Optional central cropping for consistent input sizes
- **Batch processing**: Efficient data loading with multiple workers

### Training Features
- **Flexible freezing**: Option to freeze backbone or specific layers
- **Multiple schedulers**: Cosine, step, exponential, and warmup schedulers
- **Checkpoint saving**: Regular model saving with full state preservation
- **Progress tracking**: Detailed logging and progress bars

### Self-Supervised Learning
- **DINO loss**: Implements DINO-style self-supervised contrastive learning
- **Feature extraction**: Extracts both CLS tokens and patch features
- **Projection learning**: Learns meaningful representations in 256D space

## Output Files

### Checkpoints
- `dinov3_epoch_XXX.pth`: Regular epoch checkpoints
- `dinov3_final.pth`: Final model checkpoint

### Checkpoint Contents
Each checkpoint contains:
- `epoch`: Training epoch number
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `train_loss`: Training loss
- `val_loss`: Validation loss (if available)
- `config`: Training configuration

## Usage Examples

### Basic Finetuning
```bash
# Edit config/finetune.yaml with your paths
python scripts/finetune_dinov3.py -cfg config/finetune.yaml
```

### Freeze Backbone (Transfer Learning)
```yaml
freeze_backbone: true  # Only finetune final layers
```

### Freeze First N Layers
```yaml
freeze_layers: 6  # Freeze first 6 transformer layers
```

### Custom Learning Rate Schedule
```yaml
scheduler_type: cosine_warmup
scheduler_warmup_epochs: 10
scheduler_min_lr: 1e-6
```

### Test Model
```bash
python scripts/load_finetuned_model.py \
    -model ./finetune_checkpoints/dinov3_final.pth \
    -image /path/to/test.tif \
    -model_dir /path/to/original/dinov3 \
    -device cuda
```

## Tips for Best Results

1. **Data Quality**: Ensure your TIFF images are high quality and representative
2. **Batch Size**: Use larger batch sizes if you have sufficient GPU memory
3. **Learning Rate**: Start with 1e-4 and adjust based on training dynamics
4. **Freezing Strategy**: 
   - Use `freeze_backbone: true` for quick adaptation
   - Use `freeze_layers: 6` for balanced finetuning
   - Use no freezing for full finetuning (requires more data)
5. **Crop Size**: Use 224 for faster training, 256+ for better quality
6. **Validation**: Always use validation data to monitor overfitting

## Troubleshooting

### Common Issues
1. **CUDA OOM**: Reduce batch_size or crop_size
2. **Slow Training**: Increase num_workers or use smaller crop_size
3. **Poor Convergence**: Adjust learning rate or try different freezing strategies
4. **Memory Issues**: Use gradient checkpointing or reduce model size

### Performance Tips
- Use mixed precision training (can be enabled in the script)
- Use multiple GPUs with DataParallel
- Preprocess images offline for faster loading
- Use SSD storage for faster data loading
