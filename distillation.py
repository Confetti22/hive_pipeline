"""
adapt your above best-practice distillation python implemtation script to the following details: 
1. the teacher model is dinov3-vits16, and I have its weights in this dir:
"model_dir = "/home/confetti/e5_workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m"# ViT-S/16 (patch=16)"
2. the student model is either a light_weight_user_defined cnn_based model, this model is initialized by :
"seg_model= build_semantic_seg_model(args).to(device)"
or a light-weight vit based model TinyVit
3. the the distillation training constrain is token feature distilation(representations) by Feature mimic (cosine)
Take K tap-points across the backbone (early/mid/late).(the 2/6/11 layer from dinov3-vits16 backbone, the resanable layer at student model layer)After aligning shapes with an adapter, use cosine loss the fore the features to be similar.
also, and an optinal affinity loss as in your best-practice distillation code
4.Precompute teacher tokens: add a caching layer (dict keyed by image id) to avoid teacher forward on every step.
5.the input data is gray image of uint16, tiff,of shape 512*512, and a singel 512*512 image contains multiple adjacent brain regions, is there any useful strategy to learn each brain-region semantic representation and also the relative spatial relationship of these regions?
6. the backbone of semantice_seg_model is defined as :
class semantic_seg(nn.Module):
    def __init__(self, in_channel,out_channel,filters, kernel_size,dims,mlp_filters, 
                pad_mode='reflect', act_mode='elu', norm_mode='gn', block_type='double',downsample_strategy='max_pool'):
        super().__init__()
        kwargs ={
            'in_channel': in_channel, 
            'out_channel': out_channel,
            'filters':filters, 
            'kernel_size': kernel_size, 
            'dims':dims,                 
            'pad_mode':pad_mode, 
            'act_mode':act_mode,
            'norm_mode':norm_mode, 
            'block_type':block_type,
            'downsample_strategy': downsample_strategy,
        }
        self.cnn_module = BaseAutoEncoderND_1(**kwargs)
        self.mlp_module = ConvMLP(mlp_filters,dims,l2_norm=False)

    def forward(self, x):
        bottle_neck,cnn_out = self.cnn_module(x) # B*C*H*W --> B*H*W*C --> (B*H*W)*C
        mlp_out = self.mlp_module(cnn_out)
        return bottle_neck,mlp_out
"""

from __future__ import annotations
import math, os, hashlib
from typing import List, Dict, Tuple, Optional

import tifffile as tiff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# ========= 1) Data: uint16 grayscale TIFF -> float32 tensor =========
class GrayTiffDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.paths = list(paths)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = tiff.imread(path)  # HxW, uint16
        # to float32 [0,1]
        x = torch.from_numpy(img.astype("float32")) / 65535.0  # [H,W]
        x = x.clamp(0, 1)
        x = x.unsqueeze(0)  # [1,H,W]
        image_id = path  # use path as cache key
        return x, image_id

def to_rgb_for_vit(x_gray: torch.Tensor) -> torch.Tensor:
    # x_gray: [B,1,512,512] in [0,1]
    return x_gray.repeat(1, 3, 1, 1)  # replicate channels

# ========= 2) Teacher: DINOv3 ViT-S/16 from local dir with tap hooks =========
class ViTTapHook:
    """Hook patch tokens after selected transformer blocks (timm-style ViT)."""
    def __init__(self, vit_model: nn.Module, tap_blocks_0based: List[int]):
        self.vit = vit_model
        self.taps = set(tap_blocks_0based)
        self.tokens: Dict[int, torch.Tensor] = {}
        self.handles = []
        for i, blk in enumerate(self.vit.blocks):
            if i in self.taps:
                self.handles.append(blk.register_forward_hook(self._hook_block(i)))

    def _hook_block(self, idx):
        def fn(module, inp, out):
            # out: [B, N+1, C] (cls + patches)
            self.tokens[idx] = out[:, 1:, :]  # [B, N, C]
        return fn

    def clear(self):
        self.tokens.clear()

    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()

class TeacherDinoV3(nn.Module):
    """
    Wrap your local DINOv3-ViT-S/16. If you have Dinov3HFBackbone, plug it here.
    Expect an attribute `.blocks` and output sequence tokens [B, N+1, C].
    """
    def __init__(self, model_dir: str):
        super().__init__()
        # Option A: timm placeholder (adjust if you have an HF loader)
        # self.vit = timm.create_model("vit_small_patch16_224", pretrained=True)
        # Option B: your HF-based loader
        from transformers import AutoModel
        self.vit = AutoModel.from_pretrained(
            model_dir, local_files_only=True, output_hidden_states=True
        )
        # Ensure it returns last hidden state; for HF ViT-like, `last_hidden_state`
        # You might need a small wrapper if outputs differ.

        self.embed_dim = self.vit.config.hidden_size  # 384 for ViT-S
        for p in self.parameters(): p.requires_grad_(False)
        self.eval()

    def forward_tokens(self, x_rgb: torch.Tensor,
                       tap_blocks_0based: List[int]) -> List[torch.Tensor]:
        """
        Return list[tap] of patch tokens [B, N, C_t].
        Works for HF ViT-style outputs (hidden states per layer).
        """
        with torch.no_grad():
            out = self.vit(x_rgb, output_hidden_states=True)
            # HF returns hidden_states: tuple(layer0..layerL), each [B, N+1, C]
            hs = out.hidden_states  # length L+1 incl. embeddings
            # Map your requested blocks to HF indices:
            # If encoder has 12 blocks, hidden_states index k+1 ~ after block k
            tokens_list = []
            for blk in tap_blocks_0based:
                h = hs[blk + 1]  # [B, N+1, C]
                tokens_list.append(h[:, 1:, :])  # patch tokens
            return tokens_list  # [ [B,N,C], ... ]

# ========= 3) Student: CNN or TinyViT =========
def tokens_from_cnn_bottleneck(bottleneck: torch.Tensor) -> torch.Tensor:
    # bottleneck: [B, C, H, W]  -> tokens: [B, N=H*W, C]
    B, C, H, W = bottleneck.shape
    return bottleneck.permute(0, 2, 3, 1).reshape(B, H * W, C)

class TinyViTWithTaps(nn.Module):
    def __init__(self, name="vit_tiny_patch16_224", pretrained=False):
        super().__init__()
        self.vit = timm.create_model(name, pretrained=pretrained)
        self.embed_dim = self.vit.embed_dim
        self.handles = []
        self.tokens: Dict[int, torch.Tensor] = {}

    def register_taps(self, tap_blocks_0based: List[int]):
        # remove old
        for h in self.handles: h.remove()
        self.handles.clear()
        self.tokens.clear()
        for i, blk in enumerate(self.vit.blocks):
            if i in set(tap_blocks_0based):
                self.handles.append(blk.register_forward_hook(self._hook_block(i)))

    def _hook_block(self, idx):
        def fn(module, inp, out):
            self.tokens[idx] = out[:, 1:, :]  # [B,N,C]
        return fn

    def forward_tokens(self, x: torch.Tensor,
                       tap_blocks_0based: List[int]) -> List[torch.Tensor]:
        self.register_taps(tap_blocks_0based)
        self.tokens.clear()
        _ = self.vit(x)  # run; hooks populate tokens
        return [self.tokens[i] for i in tap_blocks_0based]

# ========= 4) Losses: Feature mimic (cosine) + optional affinity =========
class FeatureMimicCosine(nn.Module):
    def __init__(self, mode: str = "vit2vit"):
        super().__init__()
        self.mode = mode  # "vit2vit" or "vit2cnn"

    def _normalize(self, s: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # s,t: [B,N,C]
        s_ = F.layer_norm(s, s.shape[-1:])
        t_ = F.layer_norm(t, t.shape[-1:])
        if self.mode == "vit2cnn":
            s_ = F.normalize(s_, dim=-1)
            t_ = F.normalize(t_, dim=-1)
        return s_, t_

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s, t = self._normalize(s, t.detach())
        cos = F.cosine_similarity(s, t, dim=-1)  # [B,N]
        return (1.0 - cos).mean()

def local_affinity_profile(x: torch.Tensor, anchors=64, window=7) -> torch.Tensor:
    # x: [B,N,C] (assume square grid)
    B, N, C = x.shape
    H = W = int(math.sqrt(N)); assert H * W == N
    x = F.layer_norm(x, x.shape[-1:])
    x = F.normalize(x, dim=-1)
    idx = torch.randperm(N, device=x.device)[:min(anchors, N)]
    hh, ww = idx // W, idx % W
    rad = window // 2
    grid = (torch.arange(H, device=x.device)[:, None] * W
            + torch.arange(W, device=x.device)[None, :])
    sims = []
    for i in range(idx.numel()):
        h0, w0 = hh[i].item(), ww[i].item()
        hs = slice(max(0, h0 - rad), min(H, h0 + rad + 1))
        ws = slice(max(0, w0 - rad), min(W, w0 + rad + 1))
        neigh = grid[hs, ws].reshape(-1)  # [M]
        anc = x[:, h0 * W + w0, :].unsqueeze(1)   # [B,1,C]
        vecs = x[:, neigh, :]                     # [B,M,C]
        sim = (anc * vecs).sum(-1).mean(dim=1)    # [B]
        sims.append(sim)
    return torch.stack(sims, dim=1).mean()  # scalar per batch

class AffinityLoss(nn.Module):
    def __init__(self, anchors=64, window=7):
        super().__init__()
        self.anchors, self.window = anchors, window

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Compare aggregated local cosine profiles (cheap & stable)
        with torch.no_grad():
            t_prof = local_affinity_profile(t.detach(), self.anchors, self.window)
        s_prof = local_affinity_profile(s, self.anchors, self.window)
        return torch.abs(s_prof - t_prof)

# ========= 5) Distiller (supports CNN student or TinyViT student) =========
class Distiller(nn.Module):
    def __init__(self,
                 teacher_dir: str,
                 student_type: str = "cnn",  # "cnn" or "tinyvit"
                 student_cnn_builder=None,   # function(args)->model if cnn
                 student_tinyvit_name: str = "vit_tiny_patch16_224",
                 taps_teacher_1based=(2, 6, 11),
                 taps_student_1based=(2, 6, 11),
                 lambda_feat=1.0, lambda_aff=0.25,
                 use_spatial_coords: bool = False):
        super().__init__()
        self.teacher = TeacherDinoV3(teacher_dir)
        self.taps_t = [k - 1 for k in taps_teacher_1based]  # 0-based
        self.student_type = student_type
        self.lambda_feat, self.lambda_aff = lambda_feat, lambda_aff
        self.use_spatial_coords = use_spatial_coords

        if student_type == "cnn":
            assert student_cnn_builder is not None, "Provide student_cnn_builder(args)->model"
            self.student = student_cnn_builder()
            # Expect (bottleneck, mlp_out) in forward; use bottleneck as feature map
            # We’ll infer student embed dim from bottleneck channels at runtime.
            self.feat_mode = "vit2cnn"
            self._student_embed_dim = None  # lazy
        else:
            self.student = TinyViTWithTaps(student_tinyvit_name, pretrained=False)
            self.student.register_taps([k - 1 for k in taps_student_1based])
            self._student_embed_dim = self.student.embed_dim
            self.feat_mode = "vit2vit"

        self.adapter: Optional[nn.Linear] = None  # built lazily
        self.coord_proj: Optional[nn.Linear] = None  # built lazily (2->Ct)
        self.feat_loss = FeatureMimicCosine(mode=self.feat_mode)
        self.aff_loss = AffinityLoss(anchors=64, window=7)

        # Simple in-memory cache: image_id -> list of tokens per tap
        self.cache: Dict[str, List[torch.Tensor]] = {}

    def _maybe_build_adapter(self, student_dim: int):
        tdim = self.teacher.embed_dim
        if self.adapter is None:
            if student_dim == tdim:
                self.adapter = nn.Identity()
            else:
                self.adapter = nn.Linear(student_dim, tdim, bias=False)

    def _maybe_build_coord_proj(self):
        # Build projection from 2D coords -> teacher embed dim
        tdim = self.teacher.embed_dim
        if self.coord_proj is None:
            self.coord_proj = nn.Linear(2, tdim, bias=False)

    @staticmethod
    def _add_token_coords(toks: torch.Tensor) -> torch.Tensor:
        """
        toks: [B,N,C]; add normalized 2D coordinates per token as features before projection.
        Returns [B,N,2] coordinate features to be projected by coord_proj outside.
        """
        B, N, _ = toks.shape
        H = W = int(math.sqrt(N)); assert H * W == N, "Tokens are expected to form a square grid"
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=toks.device),
            torch.linspace(-1, 1, W, device=toks.device),
            indexing="ij"
        )
        coords = torch.stack([xx, yy], dim=-1).reshape(1, N, 2).repeat(B, 1, 1)  # [B,N,2]
        return coords

    def forward_teacher_tokens(self, x_rgb: torch.Tensor, image_ids: List[str]) -> List[List[torch.Tensor]]:
        """
        Return per-sample list of tap tokens [ [B,N,C_t] per tap ] using cache.
        """
        B = x_rgb.size(0)
        per_sample_tokens: List[List[torch.Tensor]] = []
        for i in range(B):
            key = image_ids[i]
            if key in self.cache:
                toks = self.cache[key]
            else:
                toks = self.teacher.forward_tokens(x_rgb[i:i+1], self.taps_t)  # list of [1,N,C_t]
                toks = [t.squeeze(0) for t in toks]  # [N,C_t] each
                self.cache[key] = toks
            per_sample_tokens.append([t.unsqueeze(0) for t in toks])  # add batch dim back
        # Now we need list per tap: stack by batch
        # Transpose list-of-lists: [B][K] -> [K][B]
        K = len(self.taps_t)
        tapwise: List[List[torch.Tensor]] = [[] for _ in range(K)]
        for b in range(B):
            for k in range(K):
                tapwise[k].append(per_sample_tokens[b][k])
        # Stack per tap -> [B,N,C_t]
        tapwise = [torch.cat(tlist, dim=0) for tlist in tapwise]
        return tapwise  # length K

    def forward_student_tokens(self, x_in: torch.Tensor) -> List[torch.Tensor]:
        if self.student_type == "cnn":
            bottleneck, _ = self.student(x_in)  # [B,Cs,H',W']
            B, Cs, H, W = bottleneck.shape
            self._maybe_build_adapter(Cs)
            toks = tokens_from_cnn_bottleneck(bottleneck)  # [B,N,Cs]
            toks = self.adapter(toks)  # -> [B,N,Ct]
            if self.use_spatial_coords:
                self._maybe_build_coord_proj()
                coord_feats = self._add_token_coords(toks)           # [B,N,2]
                toks = toks + self.coord_proj(coord_feats)            # additive spatial bias
            return [toks for _ in self.taps_t]  # reuse same stage for simplicity
        else:
            toks_list = self.student.forward_tokens(x_in, [k for k in self.taps_t])  # each [B,N,Cs]
            self._maybe_build_adapter(self.student.embed_dim)
            toks_list = [self.adapter(t) for t in toks_list]  # map Cs->Ct if needed
            if self.use_spatial_coords:
                self._maybe_build_coord_proj()
                out_list: List[torch.Tensor] = []
                for t in toks_list:
                    coord_feats = self._add_token_coords(t)          # [B,N,2]
                    out_list.append(t + self.coord_proj(coord_feats))
                toks_list = out_list
            return toks_list

    def compute_losses(self, s_list: List[torch.Tensor], t_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # s_list/t_list: lists (length K) of [B,N,Ct]
        K = len(s_list)
        L_feat = 0.0
        L_aff = 0.0
        for s, t in zip(s_list, t_list):
            L_feat = L_feat + self.feat_loss(s, t)
            L_aff  = L_aff  + self.aff_loss(s, t)
        L_feat /= K
        L_aff  /= K
        L = self.lambda_feat * L_feat + self.lambda_aff * L_aff
        return L, L_feat.detach(), L_aff.detach()

    def forward(self, x_gray: torch.Tensor, image_ids: List[str]) -> Dict[str, torch.Tensor]:
        # Prepare teacher RGB input from grayscale
        x_rgb = to_rgb_for_vit(x_gray)
        t_list = self.forward_teacher_tokens(x_rgb, image_ids)   # list K of [B,N,Ct]
        s_list = self.forward_student_tokens(x_gray)             # list K of [B,N,Ct]
        loss, lfeat, laff = self.compute_losses(s_list, t_list)
        return {"loss": loss, "L_feat": lfeat, "L_aff": laff}

# ========= 6) Training loop skeleton =========
def train_distill(
    train_paths: List[str],
    teacher_dir: str,
    student_type: str,         # "cnn" or "tinyvit"
    build_cnn_fn=None,         # lambda: build_semantic_seg_model(args).to(device)
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 5e-4,
    wd: float = 0.05,
    use_spatial_coords: bool = False,
):
    ds = GrayTiffDataset(train_paths)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    distiller = Distiller(
        teacher_dir=teacher_dir,
        student_type=student_type,
        student_cnn_builder=lambda: build_cnn_fn().to(device) if build_cnn_fn else None,
        taps_teacher_1based=(2, 6, 11),
        taps_student_1based=(2, 6, 11),
        lambda_feat=1.0,
        lambda_aff=0.25,
        use_spatial_coords=use_spatial_coords
    ).to(device)

    # Optimize **student** only
    optim = torch.optim.AdamW(
        distiller.student.parameters(),
        lr=lr, weight_decay=wd
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

    distiller.teacher.eval()
    distiller.student.train()

    for ep in range(epochs):
        for x_gray, image_ids in dl:
            x_gray = x_gray.to(device, non_blocking=True)  # [B,1,512,512]

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                out = distiller(x_gray, image_ids)
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        print(f"Epoch {ep+1:03d} | loss={loss.item():.4f} | L_feat={out['L_feat'].item():.4f} | L_aff={out['L_aff'].item():.4f}")

    return distiller.student

# ========= 7) Spatial/semantic region strategies (quick hooks you can add) =========
class CoordAdd(nn.Module):
    """Add normalized (x,y) coordinate channels for CNN students to inject spatial priors."""
    def __init__(self): super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W] -> [B,3,H,W] with coord channels
        B, _, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij"
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        return torch.cat([x, coords], dim=1)  # [B,3,H,W]

# You can wrap your build_semantic_seg_model(args) to accept CoordAdd at the input,
# or add it just before feeding x into the CNN student during forward_student_tokens.
def build_cnn_fn():
    # your builder
    return build_semantic_seg_model(args).to(device)


teacher_dir = "/home/confetti/e5_workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m"

student = train_distill(train_paths, teacher_dir, student_type="cnn", build_cnn_fn=build_cnn_fn)

student = train_distill(train_paths, teacher_dir, student_type="tinyvit")