from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """
    Flexible patch embedding layer that converts images to patches.
    Supports variable input sizes without constraints.
    """
    def __init__(self, img_size: Optional[int] = None, patch_size: int = 16, 
                 in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Use Conv2d for patch embedding - this naturally handles variable input sizes
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            patches: Patch embeddings of shape (B, num_patches, embed_dim)
            grid_size: Tuple of (H_patches, W_patches)
        """
        B, C, H, W = x.shape
        
        # Apply patch embedding
        x = self.proj(x)  # (B, embed_dim, H_patches, W_patches)
        
        # Get grid dimensions
        H_patches, W_patches = x.shape[2], x.shape[3]
        
        # Flatten spatial dimensions and transpose for transformer format
        x = x.flatten(2).transpose(1, 2)  # (B, H_patches * W_patches, embed_dim)
        
        return x, (H_patches, W_patches)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, 
                 out_features: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, 
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FlexiblePositionalEncoding(nn.Module):
    """
    Flexible positional encoding that adapts to different input sizes.
    Uses learnable positional embeddings that can be interpolated.
    """
    
    def __init__(self, embed_dim: int, max_size: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_size = max_size
        
        # Create learnable positional embeddings for maximum expected size
        self.pos_embed = nn.Parameter(torch.zeros(1, max_size, embed_dim))
        
    def forward(self, x: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, num_patches, embed_dim)
            grid_size: Tuple of (H_patches, W_patches)
            
        Returns:
            x with positional embeddings added
        """
        B, N, C = x.shape
        H_patches, W_patches = grid_size
        
        # Interpolate positional embeddings to match current grid size
        if N <= self.max_size:
            # Use existing embeddings if within max size
            pos_embed = self.pos_embed[:, :N, :]
        else:
            # Interpolate for larger sizes
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),  # (1, embed_dim, max_size)
                size=N,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # (1, N, embed_dim)
        
        return x + pos_embed


class TinyViT(nn.Module):
    """
    Base TinyViT implementation with flexible input size support.
    
    Architecture matches timm's vit_tiny_patch16_224:
    - Embed dim: 192
    - Num layers: 12
    - Patch size: 16x16
    - Num heads: 3
    - MLP ratio: 4.0
    """
    
    def __init__(self, 
                 img_size: Optional[int] = None,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 192,
                 depth: int = 12,
                 num_heads: int = 3,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 num_classes: int = 1000):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = None  # Will be determined dynamically
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        # Positional encoding
        self.pos_embed = FlexiblePositionalEncoding(embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Classification logits of shape (B, num_classes)
        """
        B = x.shape[0]
        
        # Patch embedding
        x, grid_size = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional encoding
        x = self.pos_embed(x, grid_size)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Global average pooling (average over all patches)
        x = x.mean(dim=1)  # (B, embed_dim)
        
        # Classification head
        x = self.head(x)
        
        return x
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns features instead of logits.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Features of shape (B, num_patches, embed_dim)
        """
        B = x.shape[0]
        
        # Patch embedding
        x, grid_size = self.patch_embed(x)
        
        # Add positional encoding
        x = self.pos_embed(x, grid_size)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x


class TinyViTGray(TinyViT):
    """
    TinyViT model optimized for grayscale (single-channel) input.
    
    This version is specifically designed for grayscale images and uses:
    - Single channel input (in_chans=1)
    - Optimized patch embedding for grayscale data
    """
    
    def __init__(self, 
                 img_size: Optional[int] = None,
                 patch_size: int = 16,
                 embed_dim: int = 192,
                 depth: int = 12,
                 num_heads: int = 3,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 num_classes: int = 1000):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,  # Single channel for grayscale
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_classes=num_classes
        )


class TinyViTRGB(TinyViT):
    """
    TinyViT model optimized for RGB (three-channel) input.
    
    This version is specifically designed for RGB images and uses:
    - Three channel input (in_chans=3)
    - Standard patch embedding for RGB data
    """
    
    def __init__(self, 
                 img_size: Optional[int] = None,
                 patch_size: int = 16,
                 embed_dim: int = 192,
                 depth: int = 12,
                 num_heads: int = 3,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 num_classes: int = 1000):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,  # Three channels for RGB
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_classes=num_classes
        )
