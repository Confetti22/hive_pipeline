import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from skimage import color, graph
from skimage.graph import RAG
from skimage.segmentation import mark_boundaries, slic

from confettii.plot_helper import grid_plot_list_imgs


def rag_mean_feature(image, labels, connectivity=2, mode='similarity', sigma=0.001):
    """
    Build a Region Adjacency Graph (RAG) whose edge weights reflect feature similarity.

    Parameters
    ----------
    image : np.ndarray
        Feature tensor of shape (H, W, C).
    labels : np.ndarray
        Integer superpixel labels with shape (H, W).
    connectivity : int
        Pixel connectivity used to construct the RAG.
    mode : str
        Either 'similarity' (Gaussian kernel) or 'distance' (Euclidean norm).
    sigma : float
        Variance term for the similarity kernel.
    """
    rag = RAG(labels, connectivity=connectivity)

    _initialize_rag_nodes(rag, image.shape[-1])
    _accumulate_region_features(rag, image, labels)
    _finalize_mean_features(rag)
    _assign_edge_weights(rag, mode, sigma)

    return rag


def segment_and_plot_from_feats(
    feats_map,
    image,
    label=None,
    rag=None,
    slic_compactness=0.3,
    rag_weight_sigma=0.01,
    n_segments=100,
    slic_iters=30,
    ncut_thresh=0.001
):
    """
    Segment an image using a feature map and visualize intermediate artifacts.

    Parameters
    ----------
    feats_map : np.ndarray
        Feature map with shape (H, W, C).
    image : np.ndarray
        Original RGB or grayscale image with shape (H, W, [3]).
    label : np.ndarray | None
        Optional pre-computed SLIC labels.
    rag : RAG | None
        Optional pre-computed RAG based on the input labels.
    slic_compactness : float
        Compactness term used by SLIC.
    rag_weight_sigma : float
        Sigma parameter controlling the similarity weighting in the RAG.
    n_segments : int
        Target number of SLIC superpixels.
    slic_iters : int
        Maximum number of iterations for SLIC.
    ncut_thresh : float
        Threshold used by the normalized cut.
    Returns
    -------
    np.ndarray
        uint16 array of shape (H, W) storing the normalized cut result.
    """
    feats_map_rescaled = _match_feature_resolution(feats_map, image.shape[:2])
    feats_map_normalized = _normalize_feature_map(feats_map_rescaled)

    # Build superpixels on the normalized feature map if none are supplied.
    if label is None:
        label = slic(
            feats_map_normalized,
            n_segments=n_segments,
            compactness=slic_compactness,
            max_num_iter=slic_iters,
            start_label=1,
            channel_axis=-1
        )

    # Reuse an externally supplied RAG when provided to avoid recomputation.
    if rag is None:
        rag = rag_mean_feature(
            feats_map_normalized,
            label,
            mode='similarity',
            sigma=rag_weight_sigma
        )

    ncut_labels = graph.cut_normalized(label, rag, thresh=ncut_thresh)
    ncut_labels = np.asarray(ncut_labels, dtype=np.uint16)
    rgb_vis = _compute_pca_visualization(feats_map_normalized)

    # Quick side-by-side view of the raw image and PCA features.
    grid_plot_list_imgs(images=[image, rgb_vis], ncols=2, fig_size=6)

    _plot_segmentation_diagnostics(image, rgb_vis, label, rag, ncut_labels)
    return ncut_labels


def _initialize_rag_nodes(rag, num_channels):
    """Populate per-node bookkeeping fields."""
    for node_id in rag:
        rag.nodes[node_id].update(
            {
                'labels': [node_id],
                'pixel count': 0,
                'total feature': np.zeros((num_channels,), dtype=np.float64),
            }
        )


def _accumulate_region_features(rag, image, labels):
    """Sum up features and pixel counts per superpixel."""
    for index in np.ndindex(labels.shape):
        current = labels[index]
        rag.nodes[current]['pixel count'] += 1
        rag.nodes[current]['total feature'] += image[index]


def _finalize_mean_features(rag):
    """Convert accumulated sums into mean feature vectors."""
    for node_id in rag:
        total = rag.nodes[node_id]['total feature']
        count = rag.nodes[node_id]['pixel count']
        rag.nodes[node_id]['mean feature'] = total / count


def _assign_edge_weights(rag, mode, sigma):
    """Attach a scalar weight to every edge based on region similarity."""
    for x, y, data in rag.edges(data=True):
        diff = rag.nodes[x]['mean feature'] - rag.nodes[y]['mean feature']
        diff_norm = np.linalg.norm(diff)
        if mode == 'similarity':
            data['weight'] = math.e ** (-(diff_norm**2) / sigma)
        elif mode == 'distance':
            data['weight'] = diff_norm
        else:
            raise ValueError(f"The mode '{mode}' is not recognised")


def _match_feature_resolution(feats_map, image_hw):
    """Resize the feature map so it spatially matches the original image."""
    if feats_map.shape[:2] == image_hw:
        return feats_map.copy()

    zoom_factors = tuple(raw / feat for raw, feat in zip(image_hw, feats_map.shape[:2]))
    return zoom(feats_map, zoom=(*zoom_factors, 1), order=1)


def _normalize_feature_map(features):
    """Map feature values into [0, 1] range to stabilize downstream steps."""
    feat_min = features.min()
    feat_max = features.max()
    denom = feat_max - feat_min

    if np.isclose(denom, 0):
        return np.zeros_like(features)

    return (features - feat_min) / denom


def _compute_pca_visualization(features):
    """Reduce feature channels to 3 dimensions for visualization."""
    h, w, c = features.shape
    if c <= 3:
        return features.copy()

    flat_feats = features.reshape(-1, c)
    pca = PCA(n_components=3)
    rgb_vis = pca.fit_transform(flat_feats).reshape(h, w, 3)
    return _normalize_feature_map(rgb_vis)


def _plot_segmentation_diagnostics(image, rgb_vis, label, rag, ncut_labels):
    """Render diagnostic plots for the segmentation pipeline."""
    fig, ax = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(24, 24))

    ax[0].set_title('Image')
    ax[0].imshow(image, cmap='gray' if image.ndim == 2 else None)

    ax[1].set_title('Superpixels')
    ax[1].imshow(mark_boundaries(rgb_vis, label, mode='inner'))

    ax[2].set_title('RAG')
    graph.show_rag(
        label,
        rag,
        rgb_vis,
        border_color='yellow',
        img_cmap='gray',
        edge_cmap='coolwarm',
        edge_width=1,
        ax=ax[2]
    )

    ax[3].set_title('Normalized Cut')
    ax[3].imshow(color.label2rgb(ncut_labels, rgb_vis, kind='avg'))

    for axis in ax:
        axis.axis('off')

    plt.tight_layout()
    plt.show()
