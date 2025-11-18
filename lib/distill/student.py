from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from .tinyvit import TinyViT, TinyViTGray, TinyViTRGB

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


def tokens_from_cnn_bottleneck(bottleneck: torch.Tensor) -> torch.Tensor:
    b, c, h, w = bottleneck.shape
    return bottleneck.permute(0, 2, 3, 1).reshape(b, h * w, c)


class TinyViTWithTaps(nn.Module):
    """
    TinyViT wrapper that supports feature extraction at multiple transformer blocks.
    Uses local TinyViT implementation with flexible input size support.
    Supports both grayscale and RGB input types.
    """
    
    def __init__(self, name: str = "vit_tiny_patch16_224", pretrained: bool = False, 
                 input_type: str = "rgb", depth: int = 12, embed_dim: int = 192, 
                 num_heads: Optional[int] = None, mlp_ratio: float = 4.0):
        super().__init__()
        
        # Auto-calculate num_heads if not specified
        if num_heads is None:
            # Use common head configurations based on embed_dim
            if embed_dim <= 192:
                num_heads = 3
            elif embed_dim <= 384:
                num_heads = 6
            elif embed_dim <= 768:
                num_heads = 12
            else:
                num_heads = 16
        
        # Determine input type and create appropriate model
        if input_type.lower() in ["gray", "grayscale", "single"]:
            self.vit = TinyViTGray(
                img_size=None,  # Flexible input size
                patch_size=16,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=0.0,
                num_classes=0  # No classification head needed
            )
            self.input_type = "grayscale"
        elif input_type.lower() in ["rgb", "color", "three"]:
            self.vit = TinyViTRGB(
                img_size=None,  # Flexible input size
                patch_size=16,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=0.0,
                num_classes=0  # No classification head needed
            )
            self.input_type = "rgb"
        else:
            raise ValueError(f"Unsupported input_type: {input_type}. Use 'grayscale' or 'rgb'")

        self.embed_dim = self.vit.embed_dim
        self.handles: List = []
        self.tokens: Dict[int, torch.Tensor] = {}

    def register_taps(self, tap_blocks_0based: List[int]):
        """Register hooks to extract features at specified transformer blocks."""
        # Remove existing hooks
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.tokens.clear()
        
        # Register new hooks
        for i, blk in enumerate(self.vit.blocks):
            if i in set(tap_blocks_0based):
                self.handles.append(blk.register_forward_hook(self._hook_block(i)))

    def _hook_block(self, idx: int):
        """Create hook function to capture block output."""
        def fn(module, inp, out):
            self.tokens[idx] = out
        return fn

    def forward_tokens(self, x: torch.Tensor, tap_blocks_0based: List[int]) -> List[torch.Tensor]:
        """
        Forward pass that extracts features at specified transformer blocks.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C=1 for grayscale or C=3 for RGB
            tap_blocks_0based: List of 0-based block indices to extract features from
            
        Returns:
            List of feature tensors, one for each specified block
        """
        self.register_taps(tap_blocks_0based)
        self.tokens.clear()
        
        # Forward pass through the model
        B = x.shape[0]
        
        # Patch embedding
        x, grid_size = self.vit.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional encoding
        x = self.vit.pos_embed(x, grid_size)
        
        # Apply transformer blocks (hooks will capture intermediate outputs)
        for block in self.vit.blocks:
            x = block(x)
        
        # Return captured tokens
        return [self.tokens[i] for i in tap_blocks_0based]


class TinyViTWithTapsGray(TinyViTWithTaps):
    """
    Convenience class for grayscale TinyViT with taps.
    """
    
    def __init__(self, name: str = "vit_tiny_patch16_224", pretrained: bool = False):
        super().__init__(name, pretrained, input_type="grayscale")


class TinyViTWithTapsRGB(TinyViTWithTaps):
    """
    Convenience class for RGB TinyViT with taps.
    """
    
    def __init__(self, name: str = "vit_tiny_patch16_224", pretrained: bool = False):
        super().__init__(name, pretrained, input_type="rgb")


class TinyViTWithTapsTimm(nn.Module):
    """
    TinyViT wrapper using timm package that supports feature extraction at multiple transformer blocks.
    Uses timm's TinyViT implementation with flexible input size support.
    Always uses RGB input (3 channels).
    """
    
    def __init__(self, name: str = "vit_tiny_patch16_224", pretrained: bool = False, 
                 input_type: str = "rgb", depth: int = 12, embed_dim: int = 192, 
                 num_heads: Optional[int] = None, mlp_ratio: float = 4.0):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm package is required for TinyViTWithTapsTimm. Install with: pip install timm")
        
        # Create timm model - always use RGB input
        self.vit = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,  # No classification head needed
            img_size=None,  # Flexible input size
        )
        
        # Always use RGB input for timm models
        self.input_type = "rgb"
        self.embed_dim = self.vit.embed_dim
        self.handles: List = []
        self.tokens: Dict[int, torch.Tensor] = {}

    def register_taps(self, tap_blocks_0based: List[int]):
        """Register hooks to extract features at specified transformer blocks."""
        # Remove existing hooks
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.tokens.clear()
        
        # Get transformer blocks
        if hasattr(self.vit, 'blocks'):
            blocks = self.vit.blocks
        elif hasattr(self.vit, 'layers'):
            blocks = self.vit.layers
        else:
            raise AttributeError("Could not find transformer blocks in timm model")
        
        # Register new hooks
        for i, blk in enumerate(blocks):
            if i in set(tap_blocks_0based):
                self.handles.append(blk.register_forward_hook(self._hook_block(i)))

    def _hook_block(self, idx: int):
        """Create hook function to capture block output."""
        def fn(module, inp, out):
            self.tokens[idx] = out[:,1:,:]
        return fn

    def forward_tokens(self, x: torch.Tensor, tap_blocks_0based: List[int]) -> List[torch.Tensor]:
        """
        Forward pass that extracts features at specified transformer blocks.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C=3 for RGB
            tap_blocks_0based: List of 0-based block indices to extract features from
            
        Returns:
            List of feature tensors, one for each specified block
        """
        self.register_taps(tap_blocks_0based)
        self.tokens.clear()
        
        # Ensure RGB input (3 channels)
        if x.shape[1] == 1:
            # Convert grayscale to RGB by repeating the channel
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 input channels, got {x.shape[1]}")
        
        # Forward pass through timm model
        _ = self.vit.forward_features(x)
        
        # Return captured tokens
        return [self.tokens[i] for i in tap_blocks_0based]


