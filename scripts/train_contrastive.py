#!/usr/bin/env python3
import argparse
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from pathlib import Path
from torchsummary import summary
import zarr
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
# ──────────────────────────────────────────────────────────────────────────────
# Project-specific helpers
# ──────────────────────────────────────────────────────────────────────────────
from helper.contrastive_train_helper import (
    load_checkpoint,
    save_checkpoint,
    log_layer_embeddings,
)
from lib.loss.cos_loss import ContrastiveLoss
from lib.datasets.contrastive_dataset import Contrastive_dataset_3d_2d
from lib.arch.ae import build_contrastive_model, load_compose_encoder_dict
from lib.core.scheduler import WarmupCosineLR 

# =============================================================================
# Utility helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse CLI and map pipeline.yaml (via pipeline.load_cfg) → training params.

    We prefer pipeline.load_cfg so derived paths and names stay consistent across
    the orchestrated steps. Mapped keys populate the returned args namespace so
    the rest of the script can use `cfg = args`.
    """
    import os, sys
    # Ensure project root is importable (so `import pipeline` works when run from scripts/)
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    p = argparse.ArgumentParser(description="Contrastive 3-D feature training")
    p.add_argument("-cfg", type=str, default='config/pipeline.yaml', help="Path to pipeline.yaml")
    p.add_argument("-ckpt", type=str, default=None, help="Checkpoint to resume")
    p.add_argument("-device", type=str, default="cuda", help="cuda | cpu | cuda:0 …")
    args = p.parse_args()

    if args.cfg:
        # Prefer loading via pipeline.py to get derived paths and consistency
        try:
            from pipeline import load_cfg as pipeline_load_cfg
            pl_cfg = pipeline_load_cfg(args.cfg)
        except Exception:
            pl_cfg = None

        if pl_cfg and 'contrastive_mlp' in pl_cfg and 'paths' in pl_cfg:
            cm = pl_cfg['contrastive_mlp']

            args.__dict__.update(cm)
            args.__dict__.update(pl_cfg['_run'])
            args.ae_weight_path = pl_cfg['paths']['ae_weight_path']
            args.input_image = pl_cfg['paths']['input_image']

    return args


# =============================================================================
# Training helpers
# =============================================================================

from collections import defaultdict
from functools import partial
from typing import Sequence, Tuple, Union, Literal, List
Arr   = Union[np.ndarray, torch.Tensor]
Array = Union[Arr, Sequence[Arr]]   # single array or list/tuple of arrays

FEATURE_STORE = defaultdict(list)      # {layer_name: [Tensor, ...]}
HOOK_HANDLES  = []                     # so we can remove them cleanly
LAYER_ORDER   = []

def _hook(layer_name, module, inp, out):
    """
    out : Tensor shape [B, C, D, H, W] or [B, C, H, W]
    We keep it on CPU to avoid GPU memory churn.
    """
    FEATURE_STORE[layer_name].append(out.detach().cpu())

def register_hooks(model, prefix=""):
    """
    Recursively register a forward hook on *leaf* modules that have weights.
    The prefix guarantees unique names.
    """
    for name, m in model.named_children():
        full_name = f"{prefix}{name}"
        # is leaf (= no children) AND has parameters → treat as a layer of interest
        if sum(1 for _ in m.children()) == 0 and sum(p.numel() for p in m.parameters()) > 0:
            LAYER_ORDER.append(full_name)  # <-- capture order
            HOOK_HANDLES.append(m.register_forward_hook(partial(_hook, full_name)))
        else:
            register_hooks(m, f"{full_name}.")



def proxy_accuracy(pos_cos: torch.Tensor, neg_cos: torch.Tensor) -> float:
    """Fraction where positive similarity > negative (just a rough metric)."""
    return (pos_cos > neg_cos).float().mean().item()



def valid_from_roi(model, epoch, eval_data, writer):
    """Evaluate a model on a list of ROIs.

    Works with feature tensors of shape:
        • (C, H, W)                      – old behaviour
        • (C, D, H, W)                  – channel-first 3-D
        • (D, H, W, C)                  – channel-last 3-D
    For 3-D inputs, the middle depth slice (z = D//2) is used for PCA/t-SNE.
    """
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx, data_dic in enumerate(eval_data):
        roi      = data_dic['img']          # numpy (H,W) or (D,H,W)
        label    = data_dic['label']
        if len(label.shape)==2:
            label = label[np.newaxis,:]

        inp = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0).float().to(device)
        _ = model(inp).detach().cpu().numpy().squeeze() # np.ndarray

        #current impl does not support label for different roi
        log_layer_embeddings(
            FEATURE_STORE,
            writer=writer,
            epoch=epoch,
            label_volume=label,  # numpy array
            layer_order=LAYER_ORDER,       # from hook registration
            max_layers=15,
            mode="both",                   # <- t-SNE + UMAP stacked
            tsne_kwargs=dict(perplexity=20),
            umap_kwargs=dict(n_neighbors=30, min_dist=0.05,random_state=42,),
        )
        


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    *,
    n_views: int,
    pos_weight_ratio: float,
    only_pos: bool,
):
    model.train()
    run_loss =  0.0
    pos_cos_loss=  0.0
    neg_cos_loss =  0.0
    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", leave=False)):
        batch = torch.cat(batch, dim=0).to(device)  # [B*n_views, C]
        optimizer.zero_grad()
        feats = model(batch).squeeze()
        loss, pos_cos, neg_cos = ContrastiveLoss(
            features=feats,
            n_views=n_views,
            pos_weight_ratio=pos_weight_ratio,
            only_pos=only_pos,
        )
        loss.backward()
        optimizer.step()

        run_loss += loss.item()
        pos_cos_loss +=pos_cos.item()
        neg_cos_loss +=neg_cos.item()

    n_steps = len(loader)
    writer.add_scalar("train/loss", run_loss / n_steps, epoch)
    writer.add_scalar("train/pos_cos", pos_cos_loss/ n_steps, epoch)
    writer.add_scalar("train/neg_cos", neg_cos_loss/ n_steps, epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
    print(f"[Epoch {epoch}] loss={run_loss / n_steps:.4f}, pos_cos={pos_cos_loss / n_steps:.4f}, neg_cos={neg_cos_loss / n_steps:.4f}")



@torch.no_grad()
def validate(model: nn.Module, cmpsd_model: nn.Module, eval_data,
             device: torch.device, epoch: int, writer: SummaryWriter, *,
             cnn_ckpt: Path, dims,):
    #discard the last eval layer embeddings
    FEATURE_STORE.clear()
    # refresh composite encoder weights
    load_compose_encoder_dict(cmpsd_model, str(cnn_ckpt), mlp_weight_dict=model.state_dict(), dims=dims)
    valid_from_roi(cmpsd_model, epoch, eval_data, writer)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # --------------- experiment folder ------------- #

    outs_dir =  args.contrastive_out_folder
    exp_name = args.contastive_exp_name
    ckpt_dir = outs_dir / "weights" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_dir = outs_dir / 'logs' / exp_name 
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.cfg, log_dir / "config.yaml")

    feats_map = zarr.open_array(str(args.zarr_path), mode="r")
    #load the whole zarr into memory
    feats_map = feats_map[0]
    print(f"Loaded zarr → {feats_map.shape= }")
    #todo enbale user defined or auto computed feats_map loading range
    #for data cover both right and left hemisphere, only use the right hemisphere feats for contrastive training
    #if the zarr feats is fitable in memory, we can load all feats into RAM to speed up training

    ds = Contrastive_dataset_3d_2d(feats_map,d_near=args.d_near,num_pairs=args.num_pairs,n_view=args.n_views,verbose=False,sample_neighbour_sphere_dims=args.sample_neighbour_sphere_dims)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True)

    # ---------------- models ----------------------- #

    from lib.arch.mlp import MLP 
    model = MLP(args.mlp_filters).to(device)

    # --------------- optim & sched ----------------- #
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    scheduler = WarmupCosineLR(optimizer,
                           warmup_epochs= 40,
                           max_epochs=args.epoch_num)

    # --------------- resume logic ------------------ #
    start_epoch = 0
    if args.ckpt:
        start_epoch = load_checkpoint(args.ckpt, model, optimizer)
        print(f"[INFO] Resumed from {args.ckpt} (next epoch = {start_epoch})")

    # ---------------- logging ---------------------- #
    writer = SummaryWriter(log_dir=log_dir)

    # ---------------- validation data -------------- #
    cnn_ckpt = args.ae_weight_path

    # ---------------- training loop ---------------- #
    n_epochs = args.epoch_num
    ckpt_every = args.save_very_epoch
    shuffle_every = args.shuffle_very_epoch
    valid_every = args.valid_very_epoch

    for epoch in range(start_epoch, n_epochs):

        train_one_epoch(model, loader, optimizer, device, epoch, writer,
                        n_views=args.n_views, pos_weight_ratio=args.pos_weight_ratio,
                        only_pos=False)
        scheduler.step()

        # reshuffle dataset
        if (epoch + 1) % shuffle_every == 0:
            ds = Contrastive_dataset_3d_2d(feats_map,d_near=args.d_near,num_pairs=args.num_pairs,n_view=args.n_views,verbose=False)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True)

        # checkpoint
        if (epoch + 1) % ckpt_every == 0 or epoch + 1 == n_epochs:
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}.pth"
            save_checkpoint({"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"[INFO] Saved checkpoint → {ckpt_path}")

    # also drop a pipeline-friendly marker under weights/best.pth
    try:
        latest = sorted(ckpt_dir.glob('epoch_*.pth'))
        if latest:
            best_ref = ckpt_dir.parent / 'best.pth'
            import shutil as _sh
            _sh.copy2(latest[-1], best_ref)
            print(f"[INFO] best.pth → {best_ref}")
    except Exception as e:
        print(f"[WARN] Could not materialize best.pth: {e}")

    # ---------------- finalize --------------------- #
    writer.close()
    print("[Done] Training complete.")

    try:
        latest = sorted(ckpt_dir.glob('Epoch_*.pth'))
        if latest:
            # 1) Write a pipeline-friendly symlink/copy as best.ckpt
            best_ckpt = Path(ckpt_dir)/'best.pth'
            shutil.copy2(latest[-1], best_ckpt)
            print(f"Wrote best.ckpt → {best_ckpt}")

            # 2) Persist absolute path of the last-epoch pth into pipeline YAML
            last_ckpt_abs = latest[-1].resolve()
            cfg_path = Path(args.cfg)
            from lib.utils.yaml_utils import update_mlp_weight_path_in_yaml 
            update_mlp_weight_path_in_yaml(cfg_path, str(last_ckpt_abs))
    except Exception as e:
        print(f"[WARN] Post-training finalization failed: {e}")


if __name__ == "__main__":
    main()
