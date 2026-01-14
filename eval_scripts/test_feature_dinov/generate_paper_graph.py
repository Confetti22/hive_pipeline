from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import tifffile as tif



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


PARENT_SCENE = "/home/confetti/data/t1779/scenes_o"
PARENT_RESULTS = PARENT_SCENE +"/results"

def _load_row(key: str, use_cfg_path: False, arch_keys=['','s_tinyvit'],PATH_MAP=None ) -> Dict[str, np.ndarray]:
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
        roi = _prep_roi(_load_image(Path(f"{PARENT_SCENE}/{key}_roi.tif")))
        label = _load_image(Path(f"{PARENT_SCENE}/{key}_label.tif"))
        pca = _prep_rgb(_load_image(Path(f"{PARENT_RESULTS}/{key}_{arch_keys[0]}_pca.tif")))
        pca_incep = _prep_rgb(_load_image(Path(f"{PARENT_RESULTS}/{key}_{arch_keys[1]}_pca.tif")))
        gt = _load_image(Path(f"{PARENT_SCENE}/{key}_gt.tif"))
        pred = _load_image(Path(f"{PARENT_RESULTS}/{key}_{arch_keys[0]}_pred.tif"))
        pred_incep = _load_image(Path(f"{PARENT_RESULTS}/{key}_{arch_keys[1]}_pred.tif"))

    return {
        "roi": roi,
        "label": label,
        "pca": pca,
        "pca_incep": pca_incep,
        "gt": gt,
        "pred": pred,
        "pred_incep": pred_incep,
    }

def _load_interactive_row():
    dir = '/home/confetti/data/t1779/interactive_step'
    roi = tif.imread(f"{dir}/3_1_roi.tif")
    label1 = tif.imread(f"{dir}/3_1_label1.tif")
    label2 = tif.imread(f"{dir}/3_1_label2.tif")
    pred1 = tif.imread(f"{dir}/3_1_pred1.tif")
    pred2 = tif.imread(f"{dir}/3_1_pred2.tif")
    return {
        "roi": roi,
        "label1": label1, 
        "label2": label2,
        "pred1": pred1,
        "pred2": pred2,
    }   




def _label_cmap(max_label: int) -> ListedColormap:
    base = mpl.colormaps.get_cmap("tab20").resampled(max(2, max_label + 1))
    return ListedColormap(base.colors)



def build_figure(output_path: Path = Path("results/paper_figure.png")) -> Path:
    """
    Generates a matplotlib figure with 1 row and 3 columns, showing ROI and overlays.
    1 col: just the roi image
    2 col: ROI image + overlayed scr1(alpha=0.7) + pred1(alpha=0.2)
    3 col: ROI image + overlayed scr2(alpha=0.7) + pred2(alpha=0.2)
    """
    row_data = _load_interactive_row()
    
    max_val = 0
    # Using label1/2 as scr1/2 and pred1/2
    for key in ("label1", "label2", "pred1", "pred2"):
        arr = row_data.get(key)
        if arr is not None:
            max_val = max(max_val, int(np.max(arr, initial=0)))
    
    max_label = max_val
    cmap = _label_cmap(max_label)

    # Make background (value 0) transparent
    cmap_colors = cmap.colors
    if len(cmap_colors) > 0:
        cmap_colors[0] = (0, 0, 0, 0)
    transparent_cmap = ListedColormap(cmap_colors)

    fig = plt.figure(figsize=(20, 7))
    grid = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1, 1, 1],
        hspace=0.08,
        wspace=0.04,
    )

    col_titles = [
        "ROI",
        "Scr1 + Pred1",
        "Scr2 + Pred2",
    ]
    
    roi = _prep_roi(row_data['roi'])
    scr1 = row_data['label1']
    pred1 = row_data['pred1']
    scr2 = row_data['label2']
    pred2 = row_data['pred2']

    # Data for each subplot
    plot_data = [
        [ # Col 1
            (roi, {'cmap': 'gray'})
        ],
        [ # Col 2
            (roi, {'cmap': 'gray'}),
            (scr1, {'cmap': transparent_cmap, 'alpha': 0.85, 'vmin': 0, 'vmax': max_label, 'interpolation':'nearest'}),
            (pred1, {'cmap': transparent_cmap, 'alpha': 0.35, 'vmin': 0, 'vmax': max_label, 'interpolation':'nearest'})
        ],
        [ # Col 3
            (roi, {'cmap': 'gray'}),
            (scr2, {'cmap': transparent_cmap, 'alpha': 0.85, 'vmin': 0, 'vmax': max_label, 'interpolation':'nearest'}),
            (pred2, {'cmap': transparent_cmap, 'alpha': 0.35, 'vmin': 0, 'vmax': max_label, 'interpolation':'nearest'})
        ]
    ]

    for i, pdata in enumerate(plot_data):
        ax = fig.add_subplot(grid[0, i])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(col_titles[i], fontdict={'fontsize': 12})
        
        for image_data, kwargs in pdata:
            if image_data is not None:
                ax.imshow(image_data, **kwargs)

    fig.tight_layout()
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {output_path}")
    return output_path


if __name__ == "__main__":
    build_figure()

#%%
#generate the pca and pred result using inceptionv3 for the rest '2_1','2_2','2_3','3_1','3_2','3_3' scenes after genrate the pca and pred result, refresh the PATH_MAP to include those scenes using "./eval_inception_feature_for_segmentation" to generate pca from inceptionv3; using "./interactive_svc_single_viewer.py" to generate pred from inceptionv3 features, choose the model to inception and using the default training parameters to get the prediction results