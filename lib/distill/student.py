from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from lib.arch.tinyvit import TinyViT, TinyViTGray, TinyViTRGB
from lib.arch.ae import make_block


from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False



class ShuffleNetConv5Hook(nn.Module):
    """
    Wraps torchvision ShuffleNetV2 x0.5, adds emb_dim=1024,
    and captures the output of `backbone.conv5` using a forward hook.
    """

    def __init__(self,  detach_hook: bool = False):
        super().__init__()

        weights = ShuffleNet_V2_X0_5_Weights.DEFAULT 
        self.backbone = shufflenet_v2_x0_5(weights=weights)

        self.embed_dim = 1024
        self.detach_hook = detach_hook

        self.conv5_feat: torch.Tensor | None = None
        self._hook_handle = self.backbone.conv5.register_forward_hook(self._save_conv5)

    def _save_conv5(self, module: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor):
        # "after conv5" output; typically shape [B, 1024, H', W']
        self.conv5_feat = out.detach() if self.detach_hook else out

    def forward(self, x: torch.Tensor, return_conv5: bool = True):
        self.conv5_feat = None  # avoid accidentally using stale features
        logits = self.backbone(x)
        if return_conv5:
            return self.conv5_feat, logits
        return logits

    def remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class simple_cnn_embed(nn.Module):
    """
    update 2025/12/24, add 
    update 2025/12/23
    adpated from lib.arch.ae.EncoderND, change return format as bottlenect, _
    """
    def __init__(self, in_channel, filters, kernel_size, dims=3,
                 pad_mode='reflect', act_mode='elu', norm_mode='gn',
                 block_type='simple',
                 downsample_strategy='conv_stride'):  # 'conv_stride' or 'max_pool'
        super().__init__()
        assert downsample_strategy in ('conv_stride', 'max_pool'), \
            "downsample_strategy must be 'conv_stride' or 'max_pool'"

        self.dim =dims 
        self.depth = len(filters)
        self.downsample_strategy = downsample_strategy

        Pool = nn.MaxPool3d if dims== 3 else nn.MaxPool2d
        Conv = nn.Conv3d if dims== 3 else nn.Conv2d
        self.embed_dim = filters[-1]
        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode
        }

        self.down_layers = nn.ModuleList()

        # ---- Stage 0: former conv_in, now a down_layer (single block, no padding) ----
        k0 = kernel_size[0]

        if self.downsample_strategy == 'conv_stride':
            stage0 = make_block(in_channel, filters[0], k0, stride=2,
                                block_type=block_type, dim=dims, trans=False,
                                shared_kwargs=self.shared_kwargs)
        else:
            stage0_block = make_block(in_channel, filters[0], k0, stride=1,
                                      block_type=block_type, dim=dims, trans=False,
                                      shared_kwargs=self.shared_kwargs)
            stage0 = nn.Sequential(stage0_block, Pool(kernel_size=2, stride=2))

        self.down_layers.append(stage0)

        # ---- Stages 1..depth-1 ----
        for i in range(self.depth - 1):
            ks = kernel_size[min(i + 1, len(kernel_size) - 1)]

            if self.downsample_strategy == 'conv_stride':
                block = make_block(filters[i], filters[i + 1], ks, stride=2,
                                   block_type=block_type, dim=dims, trans=False,
                                   shared_kwargs=self.shared_kwargs)
                stage = block
            else:
                block = make_block(filters[i], filters[i + 1], ks, stride=1,
                                   block_type=block_type, dim=dims, trans=False,
                                   shared_kwargs=self.shared_kwargs)
                if i == self.depth - 1 -1:
                    stage = block
                else:
                    stage = nn.Sequential(block, Pool(kernel_size=2, stride=2))

            self.down_layers.append(stage)


    def forward(self, x):
        for layer in self.down_layers:
            x = layer(x)
        return x, None 


def build_simple_cnn_embed():
    model = simple_cnn_embed(in_channel=3,block_type='double',filters=[16,32,64,128],kernel_size=[7,5,5,3],dims=2,downsample_strategy='conv_stride')
    return model

def buil_shufflelnet():
    model = ShuffleNetConv5Hook()
    return model

def tokens_from_cnn_bottleneck(bottleneck: torch.Tensor) -> torch.Tensor:
    b, c, h, w = bottleneck.shape
    return bottleneck.permute(0, 2, 3, 1).reshape(b, h * w, c)



def build_student_cnn(model_type='simple'):
    model_list =['depthwise','shufflnet','simple'] 
    assert model_type in model_list

    if model_type =='depthwise':
        from lib.arch.cnn import build_s_cnn
        return build_s_cnn()
    if model_type =='shufflnet':
        return buil_shufflelnet
    if model_type =='simple':
        return build_simple_cnn_embed 



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
            dynamic_img_size=True,
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


#adapter for DPT (return extracted multilayer features )
class TinyVitBackbone(nn.Module):
    def __init__(self, model: TinyViTWithTaps|TinyViTWithTapsTimm ,patch_size: int = 16,):
        super().__init__()
        self.model = model.eval()
        self.patch_size = patch_size                            
        self.embed_dim = model.embed_dim
    
    @torch.no_grad()
    def get_intermediate_layers(self, x: torch.Tensor, n):
        return self.model.forward_tokens(x,n)






