import torch
from typing import Sequence, Tuple, Union, Literal, List

from typing import Sequence, Union
import torch

import numpy as np
try:
    import scipy.ndimage as ndi
except Exception:
    ndi = None

def accuracy(
    logits_flat: torch.Tensor,          # (N, K)  or  (N,) / (N,1)
    targets_flat: torch.Tensor,         # (N,)    int labels 0/1/…/K-1
    topk: Union[int, Sequence[int]] = 1,
    threshold: float = 0.5,             # only used when logits is 1-D
) -> Union[float, list[float]]:
    """
    Classification / segmentation accuracy.

    • If logits_flat has K >= 2 columns → top-k accuracy like ImageNet.
    • If K == 1 or logits_flat is 1-D  → binary accuracy with `sigmoid`.

    Returns
    -------
    float | list[float]   Same contract as the original function.
    """
    # -------- binary case ------------------------------------------------
    if logits_flat.ndim == 1 or logits_flat.shape[1] == 1:
        # squeeze to 1-D if needed
        if logits_flat.ndim == 2:
            logits_flat = logits_flat.squeeze(1)

        with torch.no_grad():
            probs  = torch.sigmoid(logits_flat)
            preds  = (probs >= threshold).long()
            correct = (preds == targets_flat).float().sum()
            acc = (correct / targets_flat.numel()).item()
        return acc if isinstance(topk, int) else [acc]

    # -------- multi-class case (K >= 2) ----------------------------------
    if isinstance(topk, int):
        topk = (topk,)

    with torch.no_grad():
        maxk = max(topk)
        # Shape: (N, maxk) → transpose to (maxk, N)
        _, pred = logits_flat.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()                                    # (maxk, N)

        correct = pred.eq(targets_flat.unsqueeze(0).expand_as(pred))

        accs = []
        for k in topk:
            # Flatten first k rows, count correct, normalise by N
            correct_k = correct[:k].reshape(-1).float().sum()
            accs.append((correct_k / logits_flat.size(0)).item())

    return accs[0] if len(accs) == 1 else accs

# ----- segmentation metrics (confusion+surface) -----
def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if float(d) != 0.0 else float('nan')

def _extract_surface(binary_mask: np.ndarray) -> np.ndarray:
    m = binary_mask.astype(bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    if ndi is not None:
        eroded = ndi.binary_erosion(m, structure=np.ones((3, 3), dtype=bool), border_value=0)
        return m & (~eroded)
    pad = np.pad(m, 1, mode='constant', constant_values=False)
    up    = pad[:-2, 1:-1]
    down  = pad[2:,  1:-1]
    left  = pad[1:-1, :-2]
    right = pad[1:-1, 2:]
    neighbor_bg = (~up) | (~down) | (~left) | (~right)
    return m & neighbor_bg

def _pairwise_cdist(a_pts: np.ndarray, b_pts: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial.distance import cdist as _cdist
        return _cdist(a_pts, b_pts)
    except Exception:
        diff = a_pts[:, None, :] - b_pts[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))

def _directed_surface_distances(a_mask: np.ndarray, b_mask: np.ndarray) -> tuple:
    a_surf = _extract_surface(a_mask)
    b_surf = _extract_surface(b_mask)
    a_pts = np.argwhere(a_surf)
    b_pts = np.argwhere(b_surf)
    if a_pts.size == 0 or b_pts.size == 0:
        return np.array([]), np.array([])
    dmat = _pairwise_cdist(a_pts, b_pts)
    d_ab = dmat.min(axis=1)
    d_ba = dmat.min(axis=0)
    return d_ab, d_ba

def compute_distance_metrics_for_class(pred_cls: np.ndarray, gt_cls: np.ndarray) -> tuple:
    """Return (hd95, avg_hd, assd) where avg_hd == assd for symmetry."""
    d_ab, d_ba = _directed_surface_distances(pred_cls, gt_cls)
    if d_ab.size == 0 or d_ba.size == 0:
        return float('nan'), float('nan'), float('nan')
    both = np.concatenate([d_ab, d_ba])
    hd95 = float(np.percentile(both, 95))
    avg_hd = float(np.mean(both))
    assd = avg_hd
    return hd95, avg_hd, assd

def compute_per_class_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int, valid_mask: np.ndarray):
    """
    Compute per-class precision, recall, f1, dsc, iou, hd95, avg_hd, assd for a single prediction/GT pair,
    masking to valid_mask (targets >= 0)
    """
    metrics_per_class = {
        'precision': [[] for _ in range(num_classes)],
        'recall':    [[] for _ in range(num_classes)],
        'f1':        [[] for _ in range(num_classes)],
        'iou':       [[] for _ in range(num_classes)],
        'dsc':       [[] for _ in range(num_classes)],
        'hd95':      [[] for _ in range(num_classes)],
        'avg_hd':    [[] for _ in range(num_classes)],
        'assd':      [[] for _ in range(num_classes)],
    }

    p = pred[valid_mask]
    g = gt[valid_mask]
    for c in range(num_classes):
        p_c = (p == c)
        g_c = (g == c)
        tp = int(np.logical_and(p_c, g_c).sum())
        fp = int(np.logical_and(p_c, ~g_c).sum())
        fn = int(np.logical_and(~p_c, g_c).sum())
        prec = _safe_div(tp, tp + fp)
        rec  = _safe_div(tp, tp + fn)
        f1   = _safe_div(2 * prec * rec, prec + rec) if not (np.isnan(prec) or np.isnan(rec)) else float('nan')
        iou  = _safe_div(tp, tp + fp + fn)
        dsc  = _safe_div(2 * tp, 2 * tp + fp + fn)
        metrics_per_class['precision'][c].append(prec)
        metrics_per_class['recall'][c].append(rec)
        metrics_per_class['f1'][c].append(f1)
        metrics_per_class['iou'][c].append(iou)
        metrics_per_class['dsc'][c].append(dsc)
        # Distance metrics
        pred_c_full = (pred == c) & valid_mask
        gt_c_full   = (gt   == c) & valid_mask
        hd95, avg_hd, assd = compute_distance_metrics_for_class(pred_c_full, gt_c_full)
        metrics_per_class['hd95'][c].append(hd95)
        metrics_per_class['avg_hd'][c].append(avg_hd)
        metrics_per_class['assd'][c].append(assd)
    return metrics_per_class

def merge_metric_lists(a, b):
    for k in a.keys():
        for c in range(len(a[k])):
            a[k][c].extend(b[k][c])
    return a

def summarize_seg_metrics(metrics_per_class: dict, num_classes: int):
    """
    Compute per-class and overall mean/std for each metric (used for JSON logging and reporting).
    Returns (per_class_stats, overall_stats) where each is a dict of {metric: {mean, std, text}}.
    """
    import numpy as np
    def _format_mean_std(vals):
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            m = 0.0
            s = 0.0
        else:
            m = float(np.mean(arr))
            s = float(np.std(arr))
        return m, s, f"{m:.3f} ± {s:.3f}"

    metric_names = list(metrics_per_class.keys())
    per_class_stats = {}
    for c in range(num_classes):
        per_class_stats[c] = {}
        for m in metric_names:
            mean_v, std_v, fmt = _format_mean_std(metrics_per_class[m][c])
            per_class_stats[c][m] = {'mean': mean_v, 'std': std_v, 'text': fmt}
    overall_stats = {}
    for m in metric_names:
        all_vals = []
        for c in range(num_classes):
            all_vals.extend(metrics_per_class[m][c])
        mean_v, std_v, fmt = _format_mean_std(all_vals)
        overall_stats[m] = {'mean': mean_v, 'std': std_v, 'text': fmt}
    return per_class_stats, overall_stats

def format_metric_stats(per_class_stats, overall_stats, metric_names=None):
    """
    Return a string summary of stats for console/logging.
    """
    if metric_names is None:
        metric_names = list(overall_stats.keys())
    lines = []
    for m in metric_names:
        lines.append(f"{m.upper()}:")
        for c in sorted(per_class_stats.keys()):
            lines.append(f"  class_{c}: {per_class_stats[c][m]['text']}")
        lines.append(f"  overall: {overall_stats[m]['text']}")
    return '\n'.join(lines)