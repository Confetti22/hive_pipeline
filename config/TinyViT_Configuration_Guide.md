# TinyViT Model Configuration Guide

## Overview
The TinyViT model now supports configurable architecture parameters through the config file. This guide explains how to set the `embed_dim` and other parameters for optimal performance.

## Configuration Parameters

### Core Architecture Parameters

```yaml
# TinyViT Model Architecture (only used when student_type: tinyvit)
tinyvit_depth: 12        # Number of transformer layers (6-24 recommended)
tinyvit_embed_dim: 192   # Embedding dimension (96, 192, 384, 768 recommended)
tinyvit_num_heads: 3     # Number of attention heads (auto-calculated if not specified)
tinyvit_mlp_ratio: 4.0   # MLP expansion ratio (4.0 recommended)
```

## Embedding Dimension (embed_dim) Guide

### What is embed_dim?
The embedding dimension determines the size of feature vectors that represent each image patch. It's the core parameter that controls:
- **Model capacity**: Higher dimensions = more parameters = better feature learning
- **Memory usage**: Scales quadratically with embed_dim
- **Computational cost**: Higher dimensions = slower inference
- **Feature richness**: More dimensions = more expressive features

### Recommended embed_dim Values

| embed_dim | Model Size | Use Case | Memory (512×512) | Speed | Performance |
|-----------|------------|----------|------------------|-------|-------------|
| **96** | ~1.5M | Ultra-lightweight | ~2GB | Fastest | Basic features |
| **192** | ~5.8M | Lightweight (default) | ~4GB | Fast | Good balance |
| **384** | ~22M | Medium | ~8GB | Medium | High quality |
| **768** | ~87M | Large | ~16GB | Slow | Best quality |

### Choosing embed_dim Based on Your Needs

#### 🚀 **Speed Priority** → `embed_dim: 96`
```yaml
tinyvit_embed_dim: 96
tinyvit_depth: 6
tinyvit_num_heads: 3
```
- **Best for**: Real-time applications, mobile deployment
- **Trade-off**: Lower feature quality but fastest inference
- **Memory**: ~2GB for 512×512 images

#### ⚖️ **Balanced** → `embed_dim: 192` (Default)
```yaml
tinyvit_embed_dim: 192
tinyvit_depth: 12
tinyvit_num_heads: 3
```
- **Best for**: General-purpose distillation
- **Trade-off**: Good balance of speed and quality
- **Memory**: ~4GB for 512×512 images

#### 🎯 **Quality Priority** → `embed_dim: 384`
```yaml
tinyvit_embed_dim: 384
tinyvit_depth: 12
tinyvit_num_heads: 6
```
- **Best for**: High-quality feature learning
- **Trade-off**: Slower but richer features
- **Memory**: ~8GB for 512×512 images

#### 🏆 **Maximum Quality** → `embed_dim: 768`
```yaml
tinyvit_embed_dim: 768
tinyvit_depth: 12
tinyvit_num_heads: 12
```
- **Best for**: Research, best possible quality
- **Trade-off**: Slowest inference, highest memory
- **Memory**: ~16GB for 512×512 images

### Depth (tinyvit_depth) Guidelines

| Depth | Use Case | Training Time | Performance |
|-------|----------|---------------|-------------|
| **6** | Fast training | ~2x faster | Basic features |
| **12** | Standard (default) | Baseline | Good balance |
| **18** | High quality | ~1.5x slower | Rich features |
| **24** | Maximum | ~2x slower | Best features |

### Number of Heads (tinyvit_num_heads)

**Auto-calculation** (recommended):
- embed_dim ≤ 192 → 3 heads
- embed_dim ≤ 384 → 6 heads  
- embed_dim ≤ 768 → 12 heads
- embed_dim > 768 → 16 heads

**Manual setting**:
```yaml
tinyvit_num_heads: 6  # Must divide embed_dim evenly
```

### MLP Ratio (tinyvit_mlp_ratio)

**Standard**: `4.0` (recommended)
- MLP hidden size = embed_dim × 4
- embed_dim=192 → MLP hidden=768

**Lightweight**: `2.0`
- Reduces parameters by ~50%
- May hurt performance

**Rich**: `6.0`
- Increases parameters by ~50%
- May improve performance

## Configuration Examples

### Example 1: Ultra-Fast Model
```yaml
student_type: tinyvit
tinyvit_input_type: grayscale
tinyvit_depth: 6
tinyvit_embed_dim: 96
tinyvit_mlp_ratio: 2.0
```

### Example 2: Balanced Model (Default)
```yaml
student_type: tinyvit
tinyvit_input_type: grayscale
tinyvit_depth: 12
tinyvit_embed_dim: 192
tinyvit_mlp_ratio: 4.0
```

### Example 3: High-Quality Model
```yaml
student_type: tinyvit
tinyvit_input_type: grayscale
tinyvit_depth: 18
tinyvit_embed_dim: 384
tinyvit_mlp_ratio: 4.0
```

### Example 4: Research Model
```yaml
student_type: tinyvit
tinyvit_input_type: grayscale
tinyvit_depth: 24
tinyvit_embed_dim: 768
tinyvit_mlp_ratio: 4.0
```

## Performance vs. Resource Trade-offs

### Memory Usage (512×512 images)
- embed_dim=96: ~2GB GPU memory
- embed_dim=192: ~4GB GPU memory  
- embed_dim=384: ~8GB GPU memory
- embed_dim=768: ~16GB GPU memory

### Training Time
- embed_dim=96: ~4x faster than default
- embed_dim=192: Baseline (default)
- embed_dim=384: ~4x slower than default
- embed_dim=768: ~16x slower than default

### Model Size
- embed_dim=96: ~1.5M parameters
- embed_dim=192: ~5.8M parameters
- embed_dim=384: ~22M parameters
- embed_dim=768: ~87M parameters

## Recommendations by Use Case

### 🏃‍♂️ **Real-time Applications**
```yaml
tinyvit_embed_dim: 96
tinyvit_depth: 6
tinyvit_mlp_ratio: 2.0
```

### 📱 **Mobile Deployment**
```yaml
tinyvit_embed_dim: 96
tinyvit_depth: 6
tinyvit_mlp_ratio: 2.0
```

### 🖥️ **Desktop Applications**
```yaml
tinyvit_embed_dim: 192
tinyvit_depth: 12
tinyvit_mlp_ratio: 4.0
```

### 🔬 **Research/Experimentation**
```yaml
tinyvit_embed_dim: 384
tinyvit_depth: 18
tinyvit_mlp_ratio: 4.0
```

### 🏆 **Maximum Quality**
```yaml
tinyvit_embed_dim: 768
tinyvit_depth: 24
tinyvit_mlp_ratio: 4.0
```

## Tips for Choosing Parameters

1. **Start with defaults** (embed_dim=192, depth=12)
2. **If too slow**: Reduce embed_dim to 96 or depth to 6
3. **If quality insufficient**: Increase embed_dim to 384
4. **Monitor GPU memory**: Don't exceed available memory
5. **Test on your data**: Different datasets may need different settings

## Parameter Validation

The system automatically validates:
- `embed_dim` must be divisible by `num_heads`
- `depth` should be between 1-24
- `mlp_ratio` should be between 1.0-8.0
- `num_heads` is auto-calculated if not specified
