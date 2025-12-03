from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import tifffile as tif


PARENT_SCENE = Path("/home/confetti/data/t1779/scenes_o")
PARENT_RESULTS = PARENT_SCENE / "results"

# Only rows used for the figure.
PATH_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "1_1": {
        "roi": "hp_off7000_2962_4452_sieze1536_1536_12.tif",
        "mask": None,
        "label": "hp_label.tif",
        "pca": "hp_pca.tif",
        "pred": "hp_pred.tif",
        "pca_incep": "hp_pca_inception_sliding_win.tif",
        "pred_incep": "hp_predict_inception.tif",
        "gt": "1_1_gt.tif",
    },
    "1_2": {
        "roi": "vii_1536_1536_83.tif",
        "mask": None,
        "label": "7N_label.tif",
        "pca": "7N_pca.tif",
        "pred": "7N_pred.tif",
        "pca_incep": "7N_pca_inception_sliding_win.tif",
        "pred_incep": "7N_pred_inception.tif",
        "gt": "1_2_gt.tif",
    },
    "1_3": {
        "roi": "visa_1536_1536_12.tif",
        "mask": "visa_mask.tif",
        "label": "visa_label.tif",
        "pca": "visa_pca.tif",
        "pred": "visa_pred.tif",
        "pca_incep": "visa_pca_inception_sliding_win.tif",
        "pred_incep": "visa_pred_inceptionv3.tif",
        "gt": "1_3_gt.tif",
    },
    "2_1": {
        "roi": "wf_hp_1536_1536.tif",
        "mask": None,
        "label": "wf_hp_label.tif",
        "pca": "wf_hp_pca.tif",
        "pred": "wf_hp_pred.tif",
        "pca_incep": "wf_hp_pca_inceptionv3.tif",
        "pred_incep": "wf_hp_pred_inceptionv3.tif",
        "gt": "2_1_gt.tif",
    },
    "2_2": {
        "roi": "wf_viin_1536_1536.tif",
        "mask": None,
        "label": "wf_7n_label.tif",
        "pca": "wf_7n_pca.tif",
        "pred": "wf_7n_pred.tif",
        "pca_incep": "wf_7n_pca_inceptionv3.tif",
        "pred_incep": "wf_7n_pred_inceptionv3.tif",
        "gt": "2_2_gt.tif",
    },
    "2_3": {
        "roi": "wf_visa_1536_1536.tif",
        "mask": "wf_visa_mask.tif",
        "label": "wf_visa_label.tif",
        "pca": "wf_visa_pca.tif",
        "pred": "wf_visa_pred.tif",
        "pca_incep": "wf_visa_pca_inceptionv3.tif",
        "pred_incep": "wf_visa_pred_inceptionv3.tif",
        "gt": "2_3_gt.tif",
    },
    "3_1": {
        "roi": "dk_hp_roi.tif",
        "mask": None,
        "label": "dk_hp_label.tif",
        "pca": "dk_hp_pca.tif",
        "pred": "dk_hp_pred.tif",
        "pca_incep": "dk_hp_pca_inceptionv3.tif",
        "pred_incep": "dk_hp_pred_inceptionv3.tif",
        "gt": "3_1_gt.tif",
    },
    "3_2": {
        "roi": "dk_vii_roi.tif",
        "mask": None,
        "label": "dk_7N_label.tif",
        "pca": "dk_7N_pca.tif",
        "pred": "dk_7N_pred.tif",
        "pca_incep": "dk_7N_pca_inceptionv3.tif",
        "pred_incep": "dk_7N_pred_inceptionv3.tif",
        "gt": "3_2_gt.tif",
    },
    "3_3": {
        "roi": "dk_vis_roi.tif",
        "mask": "dk_vis_mask.tif",
        "label": "dk_vis_label.tif",
        "pca": "dk_vis_pca.tif",
        "pred": "dk_vis_pred.tif",
        "pca_incep": "dk_vis_pca_inceptionv3.tif",
        "pred_incep": "dk_vis_pred_inceptionv3.tif",
        "gt": "3_3_gt_tig.tif",
    },
}


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


def _load_row(key: str, use_cfg_path: False) -> Dict[str, np.ndarray]:
    if use_cfg_path:
        cfg = PATH_MAP[key]
        roi = _prep_roi(_load_image(PARENT_SCENE / cfg["roi"]))
        label = _load_image(PARENT_RESULTS / cfg["label"]) if cfg["label"] else None
        pca = _prep_rgb(_load_image(PARENT_RESULTS / cfg["pca"]))
        pca_incep = _prep_rgb(_load_image(PARENT_RESULTS / cfg["pca_incep"]))
        gt = _load_image(PARENT_SCENE / cfg["gt"])
        pred = _load_image(PARENT_RESULTS / cfg["pred"])
        pred_incep = _load_image(PARENT_RESULTS / cfg["pred_incep"])

    else:
        roi = _prep_roi(_load_image(PARENT_SCENE /f"{key}_roi.tif"))
        label = _load_image(PARENT_SCENE / f"{key}_label.tif")
        pca = _prep_rgb(_load_image(PARENT_RESULTS / f"{key}_dpt_pca.tif"))
        pca_incep = _prep_rgb(_load_image(PARENT_RESULTS /f"{key}_inception_v3_pca.tif" ))

        gt = _load_image(PARENT_SCENE / f"{key}_gt.tif")
        pred = _load_image(PARENT_RESULTS / f"{key}_dpt_pred.tif")
        pred_incep = _load_image(PARENT_RESULTS / f"{key}_inception_v3_pred.tif")


    return {
        "roi": roi,
        "label": label,
        "pca": pca,
        "pca_incep": pca_incep,
        "gt": gt,
        "pred": pred,
        "pred_incep": pred_incep,
    }


def _label_cmap(max_label: int) -> ListedColormap:
    base = mpl.colormaps.get_cmap("tab20").resampled(max(2, max_label + 1))
    return ListedColormap(base.colors)


def _max_label_value(rows: Dict[str, Dict[str, np.ndarray]]) -> int:
    max_val = 0
    for row in rows.values():
        for key in ("label", "gt", "pred", "pred_incep"):
            arr = row.get(key)
            if arr is None:
                continue
            max_val = max(max_val, int(np.max(arr, initial=0)))
    return max_val


def build_figure(output_path: Path = Path("results/paper_graph_d0_full_dilate.png")) -> Path:
    rows = {key: _load_row(key,use_cfg_path=False) for key in ("1_1","1_2","1_3","2_1","2_2","2_3","3_1","3_2","3_3")}
    # rows = {key: _load_row(key,use_cfg_path=False) for key in ("1_1","2_1","3_1")}
    max_label = _max_label_value(rows)
    cmap = _label_cmap(max_label)

    fig = plt.figure(figsize=(18, 30))
    grid = fig.add_gridspec(
        nrows=9,
        ncols=7,
        width_ratios=[1, 1, 1, 0.18, 1, 1, 1],
        hspace=0.08,
        wspace=0.04,
    )

    col_titles = [
        "1 ROI(HR) + Scribble",
        "2 PCA (Our)",
        "3 PCA (InceptionV3)",
        "4 Ground Truth",
        "5 Prediction (Our)",
        "6 Prediction (InceptionV3)",
    ]
    row_tags = ["a", "b", "c","d","e","f","g","h","i"]
    # row_tags = ["a", "b", "c"]

    for row_idx, (row_tag, key) in enumerate(zip(row_tags, ("1_1","1_2","1_3","2_1","2_2","2_3","3_1","3_2","3_3"))):
    # for row_idx, (row_tag, key) in enumerate(zip(row_tags, ("1_1","2_1","3_1"))):
        data = rows[key]
        axes = [
            fig.add_subplot(grid[row_idx, 0]),
            fig.add_subplot(grid[row_idx, 1]),
            fig.add_subplot(grid[row_idx, 2]),
            fig.add_subplot(grid[row_idx, 4]),
            fig.add_subplot(grid[row_idx, 5]),
            fig.add_subplot(grid[row_idx, 6]),
        ]

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

        axes[1].imshow(data["pca"])
        axes[2].imshow(data["pca_incep"])
        axes[3].imshow(data["gt"], cmap=cmap, vmin=0, vmax=max_label, interpolation="nearest")
        axes[4].imshow(data["pred"], cmap=cmap, vmin=0, vmax=max_label, interpolation="nearest")
        axes[5].imshow(
            data["pred_incep"], cmap=cmap, vmin=0, vmax=max_label, interpolation="nearest"
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
    build_figure()

#%%
#generate the pca and pred result using inceptionv3 for the rest '2_1','2_2','2_3','3_1','3_2','3_3' scenes after genrate the pca and pred result, refresh the PATH_MAP to include those scenes using "./eval_inception_feature_for_segmentation" to generate pca from inceptionv3; using "./interactive_svc_single_viewer.py" to generate pred from inceptionv3 features, choose the model to inception and using the default training parameters to get the prediction results
