import numpy as np
from typing import Tuple, Optional, Literal
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops


import numpy as np
from typing import Tuple

def relabel_sequential(labels: np.ndarray, background: int = 0) -> Tuple[np.ndarray, dict]:
    """
    Remap label values in `labels` to consecutive integers from 1..N.
    The background value (default 0) is kept as 0.

    Args:
        labels:
            N-dimensional integer array (e.g., 2D label image).
        background:
            Label value to treat as background; stays as 0 in output.

    Returns:
        relabeled: np.ndarray
            Array of same shape as `labels`, with new label IDs 0..N.
        mapping: dict[int, int]
            Dictionary of old_label -> new_label mappings (excluding background).

    Example:
        >>> labels = np.array([[0,1,1,5],
        ...                    [2,2,5,5],
        ...                    [0,0,7,7]])
        >>> relabeled, mapping = relabel_sequential(labels)
        >>> mapping
        {1: 1, 2: 2, 5: 3, 7: 4}
        >>> np.unique(relabeled)
        array([0, 1, 2, 3, 4])
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != background]

    mapping = {old: new for new, old in enumerate(unique_labels, start=1)}

    relabeled = np.zeros_like(labels, dtype=int)
    for old, new in mapping.items():
        relabeled[labels == old] = new

    return relabeled, mapping

def erode_labels(
    labels: np.ndarray,
    width: float,
    spacing: Optional[Tuple[float, float]] = None,
    *,
    background: int = 0,
    method: Literal["distance"] = "distance",
) -> np.ndarray:
    """
    Erode each simply connected labeled region in a 2D label image
    by the given width, without mixing labels.

    Erosion is performed per-region using a distance transform:
    a pixel is kept iff its distance to the region boundary is >= width.
    Removed pixels are set to `background`.

    Args:
        labels:
            2D integer label image of shape (H, W). Each unique value
            (except `background`) denotes one region. Regions are assumed
            simply connected.
        width:
            Erosion width. If `spacing is None`, this is in pixels.
            If `spacing=(sy, sx)` is provided, `width` is in the same
            physical units as spacing.
        spacing:
            Optional (sy, sx). Pixel size along (row, col).
            If given, distance transform uses this sampling and `width` is
            interpreted in physical units.
        background:
            Background label value. Removed pixels are set to this value.
        method:
            Currently only "distance" (Euclidean distance transform).
            Kept for future extensibility.

    Returns:
        A new label image with each region eroded by `width`.

    Notes:
        - If `width <= 0`, the input is returned unchanged.
        - If a region is smaller than `2*width` across, it may vanish.
        - This method erodes each label independently, so labels never
          bleed into each other.
        - Complexity is roughly linear in the number of region pixels,
          with a small overhead per region (operates on per-region bounding boxes).

    Example:
        >>> out = erode_labels(lbl, width=3)  # 3-pixel erosion
        >>> out_phys = erode_labels(lbl, width=5.0, spacing=(0.5, 0.5))  # 5 units
    """
    if labels.ndim != 2:
        raise ValueError(f"`labels` must be 2D, got shape {labels.shape}.")
    if width <= 0:
        return labels.copy()

    labels = np.asarray(labels)
    out = labels.copy()

    # Iterate over regions via bounding boxes for efficiency
    props = regionprops(labels)
    if not props:
        return out

    # Choose sampling for EDT
    sampling = None if spacing is None else (float(spacing[0]), float(spacing[1]))
    thr = float(width)

    for p in props:
        lab = p.label
        if lab == background:
            continue

        minr, minc, maxr, maxc = p.bbox
        sl_r = slice(minr, maxr)
        sl_c = slice(minc, maxc)

        region = (labels[sl_r, sl_c] == lab)
        if not region.any():
            continue

        # Distance to the *boundary* inside this region
        # Pixels deep inside have larger distance; boundary pixels ~0.
        dt = distance_transform_edt(region, sampling=sampling)

        # Keep only pixels at least `width` away from boundary
        keep = dt >= thr

        # Write back: where region existed but not kept -> background
        to_clear = region & (~keep)
        if to_clear.any():
            tmp = out[sl_r, sl_c]
            tmp[to_clear] = background
            out[sl_r, sl_c] = tmp

    return out

def main():
    import napari
    import tifffile as tif
    mask_vol = tif.imread("/home/confetti/data/t1779/register_data_roi/cp_mask_reduced.tif") 
    mask = mask_vol[5]
    eroded = erode_labels(mask,width=40)
    relabelled,mappings = relabel_sequential(eroded)
    viewer = napari.Viewer(ndisplay=2)
    viewer.add_labels(mask, name='ori_mask')
    viewer.add_labels(eroded,name = 'eroded')
    viewer.add_labels(relabelled, name = 'relabelled')
    napari.run()


if __name__ == "__main__":
    main()