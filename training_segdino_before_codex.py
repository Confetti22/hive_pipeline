#%%
import itertools                      
import math, os
import numpy as np
import shutil
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config.load_config import load_cfg
import json

from lib.utils.html_logger import HTMLFigureLogger
from lib.datasets.simple_segdataset import get_dataset 

from lib.core.metric import accuracy, compute_per_class_metrics, merge_metric_lists, summarize_seg_metrics, format_metric_stats

import math

from typing import Sequence, Tuple, Union, Literal, List
Arr   = Union[np.ndarray, torch.Tensor]
Array = Union[Arr, Sequence[Arr]]   # single array or list/tuple of arrays

NUM_CLASSES = 8  # default, overridden by config

# ───────────────────────── feature hooks ──────────────────────────
import numpy as np

def merge_arrays_to_grid(array1, array2, K):
    B, H, W = array1.shape
    assert array2.shape == (B, H, W), "array2 must have same shape as array1"
    K = min(K,B)

    # Get first K images from each array
    row1 = array1[:K]   # (K, H, W)
    row2 = array2[:K]   # (K, H, W)

    # Concatenate images in each row horizontally → shape: (H, K*W)
    row1_concat = np.concatenate(row1, axis=1)
    row2_concat = np.concatenate(row2, axis=1)

    # Stack the two rows vertically → shape: (2*H, K*W)
    merged_image = np.concatenate([row1_concat, row2_concat], axis=0)

    return merged_image

def print_load_result(load_result):
    missing = load_result.missing_keys
    unexpected = load_result.unexpected_keys

    if not missing and not unexpected:
        print("✅ All weights loaded successfully.")
    else:
        print("⚠️ Some weights were not loaded exactly:")
        if missing:
            print(f"   • Missing keys ({len(missing)}):\n     {missing}")
        if unexpected:
            print(f"   • Unexpected keys ({len(unexpected)}):\n     {unexpected}")

def test_function(img_logger, dataset,test_idxes, seg_model,epoch):
    """
    recored prediction on given img idxes
    """
    seg_model.eval()

    gt_maskes   = []   # list of 2-D numpy arrays
    pred_maskes = []
    label_volumes = []

    with torch.no_grad():
        for idx in test_idxes: 
            batch = dataset[idx]
            if args.recon_loss:
                (inputs, targets,recon_targets) = batch
            else:
                (inputs, targets) = batch
            inputs = inputs.unsqueeze(0)
            targets = targets.unsqueeze(0)
            inputs, targets = inputs.to(device),targets.to(device)
            logits = seg_model(inputs)  # logits: [B, C, H, W]
            # Assertions for logits shape compatibility
            assert logits.ndim == 4, f"Expected 4D logits [B,C,H,W], got {list(logits.shape)}"
            assert logits.shape[0] == inputs.shape[0], "Batch size mismatch between inputs and logits"
            assert logits.shape[-2:] == targets.shape[-2:], "Spatial size mismatch between logits and targets"
            exp_c = getattr(getattr(getattr(seg_model, 'head', None), 'scratch', None), 'output_conv', None)
            if exp_c is not None and hasattr(exp_c, 'out_channels'):
                assert logits.shape[1] == exp_c.out_channels, f"Channel mismatch, got {logits.shape[1]} vs expected {exp_c.out_channels}"
            # ---------- prediction ----------
            probs = F.softmax(logits, dim=1)                # softmax over channel C
            pred  = torch.argmax(probs, dim=1)          # [B, H, W]
            # ---------- move to CPU once ----------
            pred_np  = pred.cpu().numpy()                # [B, H, W]
            label_np = targets.cpu().numpy()
            label_volumes.append(label_np[0])
            # one 2-D slice per sample
            gt_maskes.extend(label_np)
            pred_maskes.extend(pred_np)

    num_classes = NUM_CLASSES                       # your label count
    cmap        = plt.get_cmap('nipy_spectral', num_classes)
    max_cols    = 4                       # ≤ 4 columns in the gallery

    # --- 1) merge GT + prediction for visualisation ----------------------------
    combined_imgs = []
    for gt, pred in zip(gt_maskes, pred_maskes):
        # Put GT on the left, prediction on the right
        combined = np.hstack((gt, pred))         # shape: (H, 2*W)
        combined_imgs.append(combined)

    # --- 2) build a grid -------------------------------------------------------
    n_imgs  = len(combined_imgs)
    n_cols  = min(max_cols, n_imgs)
    n_rows  = math.ceil(n_imgs / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,figsize=(4 * n_cols, 4 * n_rows),squeeze=False)

    for idx, img in enumerate(combined_imgs):
        r, c = divmod(idx, n_cols)
        axes[r, c].imshow(img, cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[r, c].set_title(f"Sample {idx}", fontsize=10)
        axes[r, c].axis("off")

    # Hide any empty cells (when images % max_cols ≠ 0)
    for idx in range(n_imgs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.tight_layout()

    img_logger.add_figure('gt/pred',fig,global_step = epoch)

    seg_model.train()
    return label_volumes

def bnd_seg_valid(img_logger, valid_loader, seg_model,epoch):
    """
    Validate segmentation model, log a GT vs prediction grid, and return summary metrics.

    Returns: (avg_valid_loss, avg_top1, avg_top3, avg_ce_loss, avg_dice_loss)
    """

    def _mean(values):
        return sum(values) / len(values) if values else 0

    def _log_gt_pred_grid(gt_masks, pred_masks, step, tag='gt/pred', num_classes=8, max_cols=4):
        if not gt_masks:
            return
        cmap = plt.get_cmap('nipy_spectral', num_classes)

        combined_imgs = [np.hstack((gt, pred)) for gt, pred in zip(gt_masks, pred_masks)]
        n_imgs = len(combined_imgs)
        n_cols = min(max_cols, n_imgs)
        n_rows = math.ceil(n_imgs / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
        for idx, img in enumerate(combined_imgs):
            r, c = divmod(idx, n_cols)
            # axes[r, c].imshow(img, cmap=cmap, vmin=1, vmax=num_classes)
            axes[r, c].imshow(img, cmap=cmap, )
            axes[r, c].set_title(f"Sample {idx}", fontsize=10)
            axes[r, c].axis("off")
        for idx in range(n_imgs, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r, c].axis("off")
        fig.tight_layout()
        img_logger.add_figure(tag, fig, global_step=epoch)

    seg_model.eval()

    valid_losses = []
    ce_losses = []
    dice_losses = []
    total_top1 = []
    total_top3 = []
    gt_maskes   = []   # list of 2-D numpy arrays
    pred_maskes = []

    # containers for per-class metric distributions
    num_classes = NUM_CLASSES
    metric_names = [
        'precision', 'recall', 'f1', 'iou', 'dsc', 'hd95', 'avg_hd', 'assd'
    ]
    per_class_values = {m: [[] for _ in range(num_classes)] for m in metric_names}

    with torch.no_grad():
        for idx ,batch in enumerate(tqdm(valid_loader)):
            (inputs, targets) = batch

            # Ensure shapes are [B, C, H, W] and [B, H, W]
            if inputs.shape[2] ==1:
                inputs = inputs.squeeze(2)
            if targets.shape[1] ==1:
                targets = targets.squeeze(1)

            inputs, targets = inputs.to(device),targets.to(device)
            logits = seg_model(inputs)  # [B, C, H, W]

            # Assertions for logits shape compatibility
            assert logits.ndim == 4, f"Expected 4D logits [B,C,H,W], got {list(logits.shape)}"
            assert logits.shape[0] == inputs.shape[0], "Batch size mismatch between inputs and logits"
            assert logits.shape[-2:] == targets.shape[-2:], "Spatial size mismatch between logits and targets"
            exp_c = getattr(getattr(getattr(seg_model, 'head', None), 'scratch', None), 'output_conv', None)
            if exp_c is not None and hasattr(exp_c, 'out_channels'):
                assert logits.shape[1] == exp_c.out_channels, f"Channel mismatch, got {logits.shape[1]} vs expected {exp_c.out_channels}"

            # Flatten all pixels: convert to channel-last then reshape
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])  # [B*H*W, C]
            targets_flat = targets.reshape(-1)

            loss, ce_loss, dice_loss = supervised_loss_fn(logits_flat, targets_flat)
            valid_losses.append(loss.item())
            ce_losses.append(ce_loss.item())
            dice_losses.append(dice_loss.item())

            # Metrics
            top1, top3 = accuracy(logits_flat, targets_flat, topk=(1, 3))
            total_top1.append(top1)
            total_top3.append(top3)

            # Predictions for metrics and optional visualization
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)  # [B, H, W], 0..C-1
            B = pred_class.shape[0]
            for b in range(B):
                tgt_b = targets[b]
                pred_b = pred_class[b]
                tgt_np = tgt_b.detach().cpu().numpy()
                pred_np = pred_b.detach().cpu().numpy()
                valid_mask_np = np.ones_like(tgt_np, dtype=bool)
                per_img_metrics = compute_per_class_metrics(pred_np, tgt_np, num_classes, valid_mask_np)
                merge_metric_lists(per_class_values, per_img_metrics)

            # Collect a few samples for GT/Pred visualisation
            if idx < VALID_M:
                pred_np_vis  = pred_class.detach().cpu().numpy()
                label_np_vis = targets.detach().cpu().numpy()
                gt_maskes.extend(label_np_vis)
                pred_maskes.extend(pred_np_vis)

    # Log GT vs prediction grid
    _log_gt_pred_grid(gt_maskes, pred_maskes, step=epoch, tag='gt/pred', num_classes=8, max_cols=4)

    # Averages
    avg_valid_loss = _mean(valid_losses)
    avg_top1 = _mean(total_top1)
    avg_top3 = _mean(total_top3)
    avg_ce_loss = _mean(ce_losses)
    avg_dice_loss = _mean(dice_losses)

    # Summarize per-class and overall stats
    per_class_stats, overall_stats = summarize_seg_metrics(per_class_values, num_classes)
    metrics_text = format_metric_stats(per_class_stats, overall_stats)

    # Determine prefix for logging based on loader identity
    tag_prefix = 'valid' if getattr(valid_loader, 'batch_size', 1) == 1 else 'train'

    # TensorBoard logging: only overall mean/std
    if 'writer' in globals() and writer is not None:
        for m, dct in overall_stats.items():
            writer.add_scalar(f"{tag_prefix}/{m}/overall_mean", dct["mean"], epoch)
            writer.add_scalar(f"{tag_prefix}/{m}/overall_std", dct["std"], epoch)

    # JSONL logging per epoch for easy Python parsing
    try:
        metrics_record = {
            "epoch": int(epoch),
            "prefix": tag_prefix,
            "overall": overall_stats,
            "per_class": per_class_stats,
        }
        metrics_path = os.path.join(model_save_dir, f"{tag_prefix}_metrics.jsonl")
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metrics_record) + "\n")
    except Exception as e:
        print(f"Warning: failed to write metrics JSONL: {e}")

    seg_model.train()
    return avg_valid_loss, avg_top1, avg_top3, avg_ce_loss, avg_dice_loss 

 
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg_path = os.environ.get('SEGDINO_CFG', 'config/segdino.yaml')
args = load_cfg(cfg_path)

if args.test_mode:
    use_ratio = 0.1
    args.exp_name = f"_{args.exp_name}"
else:
    use_ratio = 1

NUM_CLASSES = int(getattr(args, "num_classes", NUM_CLASSES))


model_save_dir = f"{args.exp_save_dir}/{args.exp_name}"
os.makedirs(model_save_dir, exist_ok=True)
shutil.copy(cfg_path, f"{model_save_dir}/config.yaml")


writer          = SummaryWriter(f'{args.exp_save_dir}/{args.exp_name}')
img_logger      = HTMLFigureLogger(args.exp_save_dir + '/' + args.exp_name, html_name="seg_valid_result.html")
test_img_logger = HTMLFigureLogger(args.exp_save_dir + '/' + args.exp_name, html_name="seg_valid_result_test.html")
train_img_logger= HTMLFigureLogger(args.exp_save_dir + '/' + args.exp_name, html_name="train_seg_valid_result.html")

from lib.arch.segdino import DPT,LinearTokenSeg,Dinov3HFBackbone
from transformers import AutoModel, AutoConfig

model_dir = "/home/confetti/e5_workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m" if  not args.e5  else '/share/home/shiqiz/workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m'
encoder_size = getattr(args, 'encoder_size', 'base')
use_linear_head = bool(getattr(args, 'use_linear_head', False))
seg_head_layers = getattr(args, 'seg_head_layers', None)

#defined the model with config and load weights
hf_backbone = AutoModel.from_pretrained(
    model_dir, local_files_only=True, output_hidden_states=True, trust_remote_code=True
).to(device).eval()

#only define the model with config
# config = AutoConfig.from_pretrained(model_dir)
# hf_backbone = AutoModel.from_config(config).to(device).train()

backbone = Dinov3HFBackbone(hf_backbone)
if use_linear_head:
    seg_model = LinearTokenSeg(backbone=backbone, nclass=NUM_CLASSES, encoder_size=encoder_size).to(device)
else:
    seg_model = DPT(encoder_size=encoder_size, nclass=NUM_CLASSES, backbone=backbone, seg_head_layers=seg_head_layers).to(device)
seg_model.train()

#freeze backbone
seg_model.lock_backbone()

print("\n","frozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if not p.requires_grad])
print("\n","unfrozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if  p.requires_grad],"\n")
    
print(seg_model)


optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, seg_model.parameters()),
    lr=args.lr_start, weight_decay=args.weight_decay
)

# from lib.core.scheduler import WarmupCosineLR
# scheduler = WarmupCosineLR(optimizer,args.lr_warmup,args.epochs)


# %% ---------- data loaders & loggers (unchanged) -----------------------------

train_batch_size = getattr(args, 'train_batch_size', None)
if train_batch_size is None:
    if args.e5:
        train_batch_size = 32
    else:
        train_batch_size = 16


VALID_BATCH_SIZE = 1 #set batch_size of valid_loader ==1 to make sure each item in feature_store is from one image 
VALID_M = 4

if args.e5:
    train_img_dir = getattr(args, 'e5_data_path_dir')
    train_msk_dir = getattr(args, 'e5_mask_path_dir')
    valid_img_dir = getattr(args, 'e5_valid_data_path_dir')
    valid_msk_dir = getattr(args, 'e5_valid_mask_path_dir')
else:
    train_img_dir = getattr(args, 'data_path_dir')
    train_msk_dir  = getattr(args, 'mask_path_dir')
    valid_img_dir  = getattr(args, 'valid_data_path_dir')
    valid_msk_dir  = getattr(args, 'valid_mask_path_dir')

train_ds = get_dataset(
    data_path_dir=train_img_dir,
    mask_path_dir=train_msk_dir,
    use_ratio=use_ratio,
    normalize=True,
    make_3ch=True,
    shift_labels_to_zero=False
)

valid_ds = get_dataset(
    data_path_dir=valid_img_dir,
    mask_path_dir=valid_msk_dir,
    use_ratio=use_ratio,
    normalize=True,
    make_3ch=True,
    shift_labels_to_zero=False
)

train_loader   = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, drop_last=False)

valid_loader   = DataLoader(valid_ds, batch_size=VALID_BATCH_SIZE, shuffle=True, drop_last=False)
# fix_valid_loader   = DataLoader(valid_ds, batch_size=VALID_BATCH_SIZE, shuffle=False, drop_last=False)
# test_idxes = [6,56,88,140]

#~~~~~~~ weighted l1 loss ~~~~~~~~#
from lib.loss.ce_dice_combo import ComboLoss
from lib.utils.loss_utils import compute_class_weights_from_dataset

class_weights = compute_class_weights_from_dataset(train_ds, num_classes=NUM_CLASSES,recon_target_flag = False)
supervised_loss_fn = ComboLoss(class_weights=class_weights, focal=args.get("use_focal", True))


# %% ---------- training loop --------------------------------------------------
from pprint import pprint
from lib.arch.ae import modify_key,delete_key

start_epoch = 0 

for epoch in tqdm(range(start_epoch,args.epochs)):
    train_loss = []
    ce_losses = []
    dice_losses = []
    total_top1 = []
    for  batch_idx,batch in enumerate(train_loader):
        inputs, targets = batch
        inputs, targets= inputs.to(device), targets.to(device)
        if inputs.shape[2] ==1:
            inputs = inputs.squeeze(2)
        if targets.shape[1] ==1:
            targets = targets.squeeze(1)

        optimizer.zero_grad()
        logits = seg_model(inputs)          # [B, C, H, W]
        # Assertions for logits shape compatibility
        assert logits.ndim == 4, f"Expected 4D logits [B,C,H,W], got {list(logits.shape)}"
        assert logits.shape[0] == inputs.shape[0], "Batch size mismatch between inputs and logits"
        assert logits.shape[-2:] == targets.shape[-2:], "Spatial size mismatch between logits and targets"
        exp_c = getattr(getattr(getattr(seg_model, 'head', None), 'scratch', None), 'output_conv', None)
        if exp_c is not None and hasattr(exp_c, 'out_channels'):
            assert logits.shape[1] == exp_c.out_channels, f"Channel mismatch, got {logits.shape[1]} vs expected {exp_c.out_channels}"

        # flatten all pixels: convert to channel-last then reshape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]) # [B*H*W, C]
        targets_flat = targets.reshape(-1)

        loss ,ce_loss, dice_loss = supervised_loss_fn(logits_flat, targets_flat)
        train_loss.append(loss.item())
        ce_losses.append(ce_loss.item())
        dice_losses.append(dice_loss.item())
        
        loss.backward()
        optimizer.step()
        
    #after one epoch, update lr, average losses among steps in this epoch
    # scheduler.step()
    avg_loss = sum(train_loss) / len(train_loss) if train_loss else 0
    avg_ce_loss = sum(ce_losses) / len(ce_losses) if ce_losses else 0
    avg_dice_loss = sum(dice_losses) / len(dice_losses) if dice_losses else 0

    # current_lr = scheduler.get_last_lr()[0]
    # writer.add_scalar('lr',scheduler.get_last_lr()[0] , epoch)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('ce_Loss/train', avg_ce_loss, epoch)
    writer.add_scalar('dice_Loss/train', avg_dice_loss, epoch)

    print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | lr={args.lr_start:.6f}")

    #validation for supervised segmentation task loss
    if epoch % args.valid_very_epoch == 0 :

        val_loss ,avg_top1, avg_top3, val_ce_loss, val_dice_loss= bnd_seg_valid(img_logger, valid_loader, seg_model, epoch=epoch )
        writer.add_scalar("Loss/valid", val_loss, epoch)
        writer.add_scalar("top1_acc/valid", avg_top1, epoch)
        writer.add_scalar("top3_acc/valid", avg_top3, epoch)
        writer.add_scalar("ce_Loss/valid", val_ce_loss, epoch)
        writer.add_scalar("dice_Loss/valid", val_dice_loss, epoch)

    if (epoch % ( args.valid_very_epoch)) == 0:
        val_loss ,avg_top1, avg_top3, val_ce_loss, val_dice_loss = bnd_seg_valid(train_img_logger, train_loader, seg_model,  epoch)
        writer.add_scalar("top1_acc/train", avg_top1, epoch)
        writer.add_scalar("top3_acc/train", avg_top3, epoch)

    if (epoch + 1) % 1 == 0:
        save_path = os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth')

        torch.save({
                'seg_model': seg_model.state_dict(),
            }, save_path)

        print(f"Saved models to {save_path}")

img_logger.finalize()
train_img_logger.finalize()
