from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import to_rgb_for_vit
from .teacher import TeacherDinoV3
from .student import TinyViTWithTaps, TinyViTWithTapsTimm, tokens_from_cnn_bottleneck
from .losses import FeatureMimicCosine, FeatureMimicMSE, AffinityLoss


class Distiller(nn.Module):
    """
    Knowledge distillation module that transfers knowledge from a teacher DINOv3 model to a student model.
    
    The main workflow:
    1. Teacher processes RGB images and extracts features at specified transformer blocks
    2. Student processes grayscale images and extracts features at corresponding blocks
    3. Features are aligned spatially and dimensionally between teacher and student
    4. Loss is computed using feature mimicry and affinity preservation
    
    Args:
        teacher_dir: Path to pretrained DINOv3 teacher model
        student_type: Type of student model ("cnn" or "tinyvit")
        student_cnn_builder: Function to build CNN student model (required if student_type="cnn")
        student_tinyvit_name: Name of TinyViT model (used if student_type="tinyvit")
        taps_teacher_1based: 1-based indices of teacher transformer blocks to extract features from
        taps_student_1based: 1-based indices of student transformer blocks to extract features from
        lambda_feat: Weight for feature mimicry loss
        lambda_aff: Weight for affinity preservation loss
        use_spatial_coords: Whether to add spatial coordinate embeddings to features
    """
    def __init__(self,
                 teacher_dir: str,
                 ckpt_path: str = None,
                 student_type: str = "cnn",
                 student_cnn_builder=None,
                 student_tinyvit_name: str = "vit_tiny_patch16_224",
                 tinyvit_input_type: str = "rgb",
                 tinyvit_implementation: str = "local",  # "local" or "timm"
                 tinyvit_depth: int = 12,
                 tinyvit_embed_dim: int = 192,
                 tinyvit_num_heads: Optional[int] = None,
                 tinyvit_mlp_ratio: float = 4.0,
                 teacher_feature_layers: List[int] = [2, 6, 11],
                 student_feature_layers: List[int] = [2, 6, 11],
                 lambda_feat: float = 1.0,
                 lambda_aff: float = 0.25,
                 use_spatial_coords: bool = False,
                 feature_loss_type: str = "cosine"):  # "cosine" or "mse"
        super().__init__()
        # Initialize teacher model (DINOv3) - frozen during training
        self.teacher = TeacherDinoV3(teacher_dir,ckpt_path)
        # Convert 1-based tap indices to 0-based for internal use
        self.taps_t = [k - 1 for k in teacher_feature_layers]
        self.taps_s = [k - 1 for k in student_feature_layers]
        
        # Store configuration parameters
        # Note: lambda_feat and lambda_aff can be configured via distill.yaml
        self.student_type = student_type
        self.lambda_feat = lambda_feat
        self.lambda_aff = lambda_aff
        self.use_spatial_coords = use_spatial_coords

        # Initialize student model based on type
        if student_type == "cnn":
            assert student_cnn_builder is not None, "Provide student_cnn_builder(args)->model"
            self.student = student_cnn_builder()
            self.feat_mode = "vit2cnn"  # Different normalization for CNN features
        else:
            # Choose TinyViT implementation
            if tinyvit_implementation.lower() == "timm":
                # Use timm TinyViT implementation
                self.student = TinyViTWithTapsTimm(
                    student_tinyvit_name, 
                    pretrained=False, 
                    input_type=tinyvit_input_type,
                    depth=tinyvit_depth,
                    embed_dim=tinyvit_embed_dim,
                    num_heads=tinyvit_num_heads,
                    mlp_ratio=tinyvit_mlp_ratio
                )
            else:
                # Use local TinyViT implementation (default)
                self.student = TinyViTWithTaps(
                    student_tinyvit_name, 
                    pretrained=False, 
                    input_type=tinyvit_input_type,
                    depth=tinyvit_depth,
                    embed_dim=tinyvit_embed_dim,
                    num_heads=tinyvit_num_heads,
                    mlp_ratio=tinyvit_mlp_ratio
                )
            self.student.register_taps(self.taps_s)
            self.feat_mode = "vit2vit"

        self.adapter: Optional[nn.Module] = None
        self.coord_proj: Optional[nn.Linear] = None
        self.spatial_adapter: Optional[nn.Module] = None
        
        # Initialize feature loss based on type
        if feature_loss_type.lower() == "mse":
            self.feat_loss = FeatureMimicMSE()
        else:  # default to cosine
            self.feat_loss = FeatureMimicCosine(mode=self.feat_mode)
        
        self.aff_loss = AffinityLoss(anchors=64, window=7)

    def _maybe_build_adapter(self, student_dim: int, device: torch.device):
        """
        Builds a linear adapter to match student feature dimensions to teacher dimensions.
        
        Args:
            student_dim: Dimension of student features
            device: Device to move the adapter to
        """
        tdim = self.teacher.embed_dim
        if self.adapter is None:
            # Use identity if dimensions already match, otherwise use linear projection
            self.adapter = nn.Identity() if student_dim == tdim else nn.Linear(student_dim, tdim, bias=False)
            # Move adapter to the correct device
            self.adapter = self.adapter.to(device)

    def _maybe_build_coord_proj(self, device: torch.device):
        """
        Builds a linear projection layer for spatial coordinate embeddings.
        
        Projects 2D spatial coordinates (x, y) to the teacher's embedding dimension.
        This allows the model to learn spatial relationships in the feature space.
        
        Args:
            device: Device to move the coordinate projection to
        """
        if self.coord_proj is None:
            self.coord_proj = nn.Linear(2, self.teacher.embed_dim, bias=False)
            # Move coordinate projection to the correct device
            self.coord_proj = self.coord_proj.to(device)

    def _maybe_build_spatial_adapter(self, student_spatial: Tuple[int, int], teacher_spatial: Tuple[int, int], device: torch.device):
        """
        Builds spatial adapter to handle differences in spatial resolution between teacher and student.
        
        Args:
            student_spatial: (height, width) of student feature maps
            teacher_spatial: (height, width) of teacher feature maps
            device: Device to move the spatial adapter to
        """
        if self.spatial_adapter is None and student_spatial != teacher_spatial:
            # Use adaptive pooling to match spatial dimensions
            self.spatial_adapter = nn.AdaptiveAvgPool2d(teacher_spatial)
            # Move spatial adapter to the correct device
            self.spatial_adapter = self.spatial_adapter.to(device)

    @staticmethod
    def _add_token_coords(toks: torch.Tensor) -> torch.Tensor:
        """
        Generates 2D spatial coordinate embeddings for each token.
        
        Creates normalized coordinates in [-1, 1] range for each spatial position.
        This helps the model understand spatial relationships between features.
        
        Args:
            toks: Token tensor of shape (batch_size, num_tokens, embed_dim)
            
        Returns:
            coords: Coordinate tensor of shape (batch_size, num_tokens, 2)
        """
        b, n, _ = toks.shape
        h = w = int(math.sqrt(n)); assert h * w == n
        # Create coordinate grid in [-1, 1] range
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=toks.device),
            torch.linspace(-1, 1, w, device=toks.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1).reshape(1, n, 2).repeat(b, 1, 1)
        return coords

    def forward_teacher_tokens(self, x_rgb: torch.Tensor, image_ids: List[str]) -> List[torch.Tensor]:
        """
        Extracts teacher features at specified transformer blocks.
        
        Teacher features are extracted from multiple transformer blocks (taps).
        
        Args:
            x_rgb: RGB input images of shape (batch_size, 3, H, W)
            image_ids: List of unique identifiers for each image (unused, kept for compatibility)
            
        Returns:
            List of teacher token tensors, one for each tap
        """
        # Extract teacher features for the entire batch
        return self.teacher.forward_tokens(x_rgb, self.taps_t)

    def forward_student_tokens(self, x_in: torch.Tensor) -> List[torch.Tensor]:
        """
        Extracts student features and aligns them with teacher features.
        
        Handles both CNN and ViT student models, applies dimension adaptation,
        spatial resolution matching, and optional spatial coordinate embeddings.
        
        Args:
            x_in: Input images of shape (batch_size, channels, H, W)
            
        Returns:
            List of student token tensors, one for each tap
        """
        device = x_in.device  # Get device from input tensor
        
        if self.student_type == "cnn":
            # Extract features from CNN student model
            bottleneck, _ = self.student(x_in)
            b, cs, h, w = bottleneck.shape
            
            # Build dimension adapter if needed
            self._maybe_build_adapter(cs, device)
            
            # Convert CNN features to token format
            toks = tokens_from_cnn_bottleneck(bottleneck)
            toks = self.adapter(toks)
            
            # Handle spatial resolution differences
            student_spatial = (h, w)
            teacher_spatial = (14, 14)  # DINOv3 ViT-S/16 default
            self._maybe_build_spatial_adapter(student_spatial, teacher_spatial, device)
            if self.spatial_adapter is not None:
                # Reshape back to spatial format for pooling
                toks_spatial = toks.reshape(b, h, w, -1).permute(0, 3, 1, 2)
                toks_spatial = self.spatial_adapter(toks_spatial)
                toks = toks_spatial.permute(0, 2, 3, 1).reshape(b, -1, toks.shape[-1])
            
            # Add spatial coordinate embeddings if enabled
            if self.use_spatial_coords:
                self._maybe_build_coord_proj(device)
                coord_feats = self._add_token_coords(toks)
                toks = toks + self.coord_proj(coord_feats)
            
            # Return same features for all taps (CNN has single bottleneck)
            return [toks for _ in self.taps_s]
        else:
            # Extract features from ViT student model
            toks_list = self.student.forward_tokens(x_in, self.taps_s)
            
            # Build dimension adapter if needed
            self._maybe_build_adapter(self.student.embed_dim, device)
            toks_list = [self.adapter(t) for t in toks_list]
            
            # Add spatial coordinate embeddings if enabled
            if self.use_spatial_coords:
                self._maybe_build_coord_proj(device)
                out_list: List[torch.Tensor] = []
                for t in toks_list:
                    coord_feats = self._add_token_coords(t)
                    out_list.append(t + self.coord_proj(coord_feats))
                toks_list = out_list
            
            return toks_list

    def compute_losses(self, s_list: List[torch.Tensor], t_list: List[torch.Tensor]):
        """
        Computes distillation losses between student and teacher features.
        
        Combines feature mimicry loss (cosine similarity) and affinity preservation loss
        (local neighborhood structure) across all transformer taps.
        
        Args:
            s_list: List of student feature tensors (one per tap)
            t_list: List of teacher feature tensors (one per tap)
            
        Returns:
            total_loss: Weighted combination of all losses
            l_feat: Detached feature mimicry loss
            l_aff: Detached affinity preservation loss
        """
        k = len(s_list)
        l_feat = torch.tensor(0.0, device=s_list[0].device, dtype=s_list[0].dtype)
        l_aff = torch.tensor(0.0, device=s_list[0].device, dtype=s_list[0].dtype)
        
        # Compute losses for each tap
        for s, t in zip(s_list, t_list):
            l_feat = l_feat + self.feat_loss(s, t)  # Feature-level similarity
            if self.lambda_aff:
                l_aff = l_aff + self.aff_loss(s, t)     # Local affinity preservation
        
        # Average across taps
        l_feat /= k
        if self.lambda_aff:
            l_aff /= k
        # Weighted combination
        lambda_feat_tensor = torch.tensor(self.lambda_feat, device=l_feat.device, dtype=l_feat.dtype)
        lambda_aff_tensor = torch.tensor(self.lambda_aff, device=l_aff.device, dtype=l_aff.dtype)
        total = lambda_feat_tensor * l_feat + lambda_aff_tensor * l_aff
        return total, l_feat.detach(), l_aff.detach()

    def forward(self, x_rgb: torch.Tensor, image_ids: List[str]) -> Dict[str, torch.Tensor]:
        """
        Main forward pass for knowledge distillation.
        
        The complete workflow:
        1. Convert grayscale input to RGB for teacher
        2. Normalize inputs with ImageNet mean/std
        3. Extract teacher features (with caching)
        4. Extract student features (with dimension/spatial alignment)
        5. Compute distillation losses
        
        Args:
            x_gray: Grayscale input images of shape (batch_size, 1, H, W)
            image_ids: List of unique identifiers for each image
            
        Returns:
            Dictionary containing total loss and individual loss components
        """
        # Convert grayscale to RGB for teacher model
        
        # Extract teacher features
        t_list = self.forward_teacher_tokens(x_rgb, image_ids)
        
        # Extract student features (with alignment)
        s_list = self.forward_student_tokens(x_rgb)
        
        # Compute distillation losses
        loss, lfeat, laff = self.compute_losses(s_list, t_list)
        
        return {"loss": loss, "L_feat": lfeat, "L_aff": laff}


