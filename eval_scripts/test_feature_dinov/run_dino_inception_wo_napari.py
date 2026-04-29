
from __future__ import annotations
import os
import sys
import numpy as np
import tifffile as tif

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from lib.utils.test_scene import get_path_map
from lib.utils.test_scene import load_t1779
from interactive_svc_single_viewer import NapariSegTool

DOWNFACTOR = 0
NAPARI = False 

SAVE_PARENT = '/home/confetti/data/t1779/contrastive_pretrained_d1' #used to save pred and pca of this run
PARENT_RESULTS = SAVE_PARENT +"/results"
os.makedirs(SAVE_PARENT, exist_ok=True)   
os.makedirs(PARENT_RESULTS, exist_ok=True)

def gennerate_pred_pca(arch_list:Optional[Sequence[str]],model_dir_list:Optional[Sequence[str]],keys=None,smooth_params=(16,4,1)):
    # for arch in ['inception_v3','dpt','s_tinyvit','s_tinyvittimm']:
    
    keys = get_path_map().keys() if keys is None else keys
    arch_list = ['dpt'] if arch_list is None else arch_list

    for arch,model_dir in zip(arch_list,model_dir_list):
        for key in keys: 
            print(f"begin processing region: {key}")
            dims=3
            seg_tool = NapariSegTool(viewer=None, dims=dims, napari=NAPARI, smooth_params=smooth_params,region_key=key,down_factor=DOWNFACTOR)
    
            gt =  seg_tool.gt 

            # build & train & eval
            seg_tool.widget_train_eval(arch = arch,model_dir=model_dir,epochs=15, batch_size=16, lr=1e-4,
                                        patch_h=1536, patch_w=1536, patch_d=32,tv_denoise_weight=10)


            tif.imwrite(f"{SAVE_PARENT}/{key}_gt.tif", (gt.astype(np.uint8)))
            tif.imwrite(f"{SAVE_PARENT}/{key}_roi.tif", (seg_tool.roi.astype(np.uint16)))
            tif.imwrite(f"{SAVE_PARENT}/{key}_label.tif", (seg_tool.label.astype(np.uint8)))
            tif.imwrite(f"{SAVE_PARENT}/{key}_mask.tif", seg_tool.mask.astype(np.uint8))

            tif.imwrite(f"{PARENT_RESULTS}/{key}_{arch}_pca.tif", (seg_tool.pca_feat*255).astype(np.uint8))
            tif.imwrite(f"{PARENT_RESULTS}/{key}_{arch}_pred.tif", seg_tool.pred.astype(np.uint8))
            del seg_tool


def _load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Expected image at {path}")
    arr = tif.imread(path)
    return np.squeeze(arr)


def _prep_roi(roi: np.ndarray) -> np.ndarray:
    if roi.ndim == 3 and roi.shape[-1] != 3:
        roi = roi.max(axis=0)
    return np.squeeze(roi)


def _prep_rgb(img: np.ndarray) -> np.ndarray:
    img = np.squeeze(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img.astype(np.float32, copy=False)
    max_val = float(img.max(initial=0.0))
    min_val = float(img.min(initial=0.0))
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    return np.clip(img, 0.0, 1.0)



def _load_row(
    key: str,
    use_cfg_path: bool = False,
    arch_keys: Optional[Sequence[str]] = None,
    PATH_MAP=None,
) -> Dict[str, np.ndarray]:
    if arch_keys is None:
        arch_keys = []
    #TODO: modify use_cfg_path logic
    if use_cfg_path:
        if PATH_MAP is None:
            PATH_MAP = get_path_map()
        cfg = PATH_MAP[key]
        roi = _prep_roi(_load_image(SAVE_PARENT / cfg["roi"]))
        label = _load_image(PARENT_RESULTS / cfg["label"]) if cfg["label"] else None
        gt = _load_image(SAVE_PARENT / cfg["gt"])
        pca_list = []
        pred_list = []
        if cfg.get("pca"):
            pca_list.append(_prep_rgb(_load_image(PARENT_RESULTS / cfg["pca"])))
        if cfg.get("pca_incep"):
            pca_list.append(_prep_rgb(_load_image(PARENT_RESULTS / cfg["pca_incep"])))
        if cfg.get("pred"):
            pred_list.append(_load_image(PARENT_RESULTS / cfg["pred"]))
        if cfg.get("pred_incep"):
            pred_list.append(_load_image(PARENT_RESULTS / cfg["pred_incep"]))

    else:
        roi = _prep_roi(_load_image(Path(f"{SAVE_PARENT}/{key}_roi.tif")))
        label = _load_image(Path(f"{SAVE_PARENT}/{key}_label.tif"))
        gt = _load_image(Path(f"{SAVE_PARENT}/{key}_gt.tif"))

        pca_list = []
        pred_list = []
        for arch in arch_keys:
            pca = _prep_rgb(_load_image(Path(f"{PARENT_RESULTS}/{key}_{arch}_pca.tif")))
            pred = _load_image(Path(f"{PARENT_RESULTS}/{key}_{arch}_pred.tif"))
            pca_list.append(pca)
            pred_list.append(pred)


    return {
        "roi": roi,
        "label": label,
        "gt": gt,
        "pca": pca_list,
        "pred": pred_list,
    }


def _label_cmap(max_label: int) -> ListedColormap:
    base = mpl.colormaps.get_cmap("tab20").resampled(max(2, max_label + 1))
    return ListedColormap(base.colors)


def _max_label_value(rows: Dict[str, Dict[str, np.ndarray]]) -> int:
    max_val = 0
    for row in rows.values():
        for key in ("label", "gt"):
            arr = row.get(key)
            if arr is None:
                continue
            max_val = max(max_val, int(np.max(arr, initial=0)))
        preds = row.get("pred")
        if preds is None:
            continue
        if isinstance(preds, list):
            for pred in preds:
                if pred is None:
                    continue
                max_val = max(max_val, int(np.max(pred, initial=0)))
        else:
            max_val = max(max_val, int(np.max(preds, initial=0)))
    return max_val


def build_figure(
    output_path: Path ,
    arch_keys: Optional[Sequence[str]] = None,
    row_keys: Optional[Sequence[str]] = None,
) -> Path:
    if arch_keys is None:
        arch_keys = ["dpt", "s_tinyvit"]
    arch_keys = list(arch_keys)
    if not arch_keys:
        raise ValueError("arch_keys must contain at least one architecture.")

    row_keys = ("1_1", "1_2", "1_3", "2_1", "2_2", "2_3", "3_1", "3_2", "3_3") if row_keys is None else row_keys
    rows = {key: _load_row(key, arch_keys=arch_keys, use_cfg_path=False) for key in row_keys}
    # rows = {key: _load_row(key,use_cfg_path=False) for key in ("1_1","2_1","3_1")}
    max_label = _max_label_value(rows)
    cmap = _label_cmap(max_label)

    n_arch = len(arch_keys)
    spacer_ratio = 0.18 if n_arch <= 2 else (0.14 if n_arch <= 4 else 0.12)
    col_width = 2.6
    row_height = 3.0
    fig_width = col_width * (2 * n_arch + 2 + spacer_ratio)
    fig_height = row_height * len(rows)
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(
        nrows=len(rows),
        ncols=2 * n_arch + 3,
        width_ratios=[1] * (n_arch + 1) + [spacer_ratio] + [1] * (n_arch + 1),
        hspace=0.08,
        wspace=0.04 if n_arch <= 2 else 0.03,
    )

    col_titles = ["ROI(HR) + Scribble"]
    col_titles.extend([f"PCA ({arch})" for arch in arch_keys])
    col_titles.append("Ground Truth")
    col_titles.extend([f"Prediction ({arch})" for arch in arch_keys])
    row_tags = [chr(ord("a") + idx) for idx in range(len(row_keys))]
    # row_tags = ["a", "b", "c"]

    for row_idx, (row_tag, key) in enumerate(zip(row_tags, row_keys)):
    # for row_idx, (row_tag, key) in enumerate(zip(row_tags, ("1_1","1_2","1_3"))):
        data = rows[key]
        axes = [fig.add_subplot(grid[row_idx, 0])]
        for idx in range(n_arch):
            axes.append(fig.add_subplot(grid[row_idx, 1 + idx]))
        gt_col = n_arch + 2
        axes.append(fig.add_subplot(grid[row_idx, gt_col]))
        pred_start = n_arch + 3
        for idx in range(n_arch):
            axes.append(fig.add_subplot(grid[row_idx, pred_start + idx]))

        roi = data["roi"]
        vmax = np.percentile(roi, 98) if roi.size else None
        axes[0].imshow(roi, cmap="gray", vmax=vmax)
        if data["label"] is not None:
            label_masked = np.ma.masked_equal(data["label"], 0)
            axes[0].imshow(
                label_masked,
                cmap=cmap,
                vmin=0,
                vmax=max_label,
                alpha=0.83,
                interpolation="nearest",
            )

        for idx, pca_img in enumerate(data["pca"]):
            axes[1 + idx].imshow(pca_img)

        gt_axis = 1 + n_arch
        axes[gt_axis].imshow(data["gt"], cmap=cmap, vmin=0, vmax=max_label, interpolation="nearest")

        for idx, pred_img in enumerate(data["pred"]):
            axes[gt_axis + 1 + idx].imshow(
                pred_img,
                cmap=cmap,
                vmin=0,
                vmax=max_label,
                interpolation="nearest",
            )

        for idx, ax in enumerate(axes):
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_titles[idx], fontsize=12, pad=6)

        pos = axes[0].get_position()
        fig.text(
            pos.x0 - 0.04,
            pos.y0 + pos.height / 2.0,
            row_tag,
            fontsize=13,
            fontweight="bold",
            va="center",
            ha="right",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    # arch_keys = ['cmpsd','inception_v3']
    arch_keys = ['cmpsd_old',]
    ckpt_list = [
        # None,
        # '/home/confetti/e5_workspace/hive1_pipeline/runs/contrastive/onestage_batch2028_nview2_infolossFalse_t1779_2um/model_epoch_100.pth',
        # '/home/confetti/e5_workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m',
        # 'runs/distill/tinyvit_aug_True_t1779/student_epoch_100.pth'
    ]

    keys = ['1_1','1_2','1_3'] 
    
    generate = True #train and predict
    draw = True #display result as a matplotlib.figure

    if generate: 
        gennerate_pred_pca(arch_list=arch_keys,model_dir_list=ckpt_list,keys=keys)
    if draw:
        build_figure(output_path=Path(f"/home/confetti/e5_workspace/hive1_pipeline/results/contrastive_two_stage_t1779.png"),  arch_keys=arch_keys,row_keys=keys)
    
