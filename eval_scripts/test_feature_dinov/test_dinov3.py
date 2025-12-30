#%%
#!/usr/bin/env python3
"""Quick visual sanity check for DINO feature tokens on histology patches."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from skimage import io
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor, AutoModel


# ---------------------------------------------------------------------------
# User-configurable defaults
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHOICE = "dinov3_vits16"  # {'dinov2_vitl14', 'dinov3_vits16'}
MODEL_REGISTRY = {
    "dinov2_vitl14": "facebook/dinov2-large",  # ViT-L/14 (patch=14)
    "dinov3_vits16": "/home/confetti/e5_workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m",
}
LOCAL_ONLY = MODEL_CHOICE == "dinov3_vits16"

BACKBONE_DEFAULT = "inception_bn"
BACKBONE_CHOICES = ("dino", "inception_bn")

LAYER_LIST = [1, 7, 9, 11, 12]
IMAGE_PATH = Path("/home/confetti/data/mousebrainatlas/histology/102117913_d0.png")
CROP_OFFSET = (1800, 1800)
CROP_SIZE = 800
RESIZE_TO = 800 
CROP_TO = 800 
UPSAMPLE_VISUAL_TO = None  # e.g., 512 to upscale for presentation
SAVE_DIR = Path("runs/test_dinv")

_DEFAULT_BLUR_SIGMA = 0.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-blur",
        type=float,
        default=_DEFAULT_BLUR_SIGMA,
        help="Gaussian blur sigma applied on the token grid (in token units).",
    )
    parser.add_argument(
        "--backbone",
        choices=BACKBONE_CHOICES,
        default=BACKBONE_DEFAULT,
        help="Choose between ViT (DINO) tokens or Inception-BN feature maps.",
    )
    return parser.parse_known_args()[0]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def load_histology_patch(image_path: Path, offset: tuple[int, int], size: int) -> np.ndarray:
    """Load the requested crop from a large histology image."""
    img = io.imread(image_path)
    r0, c0 = offset
    patch = img[r0 : r0 + size, c0 : c0 + size, :]
    print(f"Loaded patch {patch.shape} from {image_path.name}")
    return patch



def preprocess_image(image: np.ndarray, transform: transforms.Compose) -> torch.Tensor:
    tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    print(f"Preprocessed tensor shape: {tensor.shape}")
    return tensor


def resolve_patch_size(model: AutoModel) -> int:
    patch = getattr(model.config, "patch_size", None)
    if isinstance(patch, (list, tuple)):
        patch = patch[0]
    if patch is None:
        patch = 14 if "vitl14" in MODEL_CHOICE.lower() else 16
    return int(patch)


def get_n_prefix_tokens(model, total_tokens: int, H_tok: int, W_tok: int) -> int:
    """Figure out how many CLS/register tokens precede the spatial grid."""
    n_reg = getattr(model.config, "num_register_tokens", None)
    if n_reg is not None:
        return 1 + int(n_reg)
    n_prefix = total_tokens - (H_tok * W_tok)
    if n_prefix < 1:
        raise ValueError(f"Unexpected token layout: {total_tokens=} vs {H_tok * W_tok=}")
    return n_prefix

class EnsureThreeChannels:
    """Ensure input image has 3 channels.
    - If grayscale (1×H×W or H×W), duplicate to 3 channels.
    - If already 3×H×W, pass through.
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Handle H×W (no channel dim)
        if tensor.ndim == 2:  
            tensor = tensor.unsqueeze(0)

        # After ToTensor: C×H×W
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)

        return tensor


def build_preprocess(mean: list[float], std: list[float]) -> transforms.Compose:
    """
    Preprocessing pipeline that supports both grayscale and RGB images.

    Steps:
    1. Convert image to tensor in [0, 1].
    2. Ensure 3-channel input (duplicate grayscale).
    3. Resize to target resolution.
    4. Center crop to model input.
    5. Normalize with model statistics.
    """
    return transforms.Compose([
        transforms.ToTensor(),  # [0,255] → [0,1], HWC → CHW
        EnsureThreeChannels(),  # grayscale → RGB duplication
        transforms.Resize(RESIZE_TO, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(CROP_TO),
        transforms.Normalize(mean=mean, std=std),
    ])


def tokens_to_pca_rgb(x_bhwc: torch.Tensor) -> np.ndarray:
    """Project token grid [H, W, C] down to 3 PCA channels for visualization."""
    x = x_bhwc.detach().cpu().float().numpy()
    H, W, C_ = x.shape
    X_proj = PCA(n_components=3).fit_transform(x.reshape(-1, C_)).reshape(H, W, 3)
    X_proj -= X_proj.min()
    X_proj /= (X_proj.max() - X_proj.min() + 1e-8)
    return X_proj.astype(np.float32)


def upsample_rgb(arr_hw3: np.ndarray, out_size: int | None) -> np.ndarray:
    if out_size is None:
        return arr_hw3
    img_ = Image.fromarray((arr_hw3 * 255).astype(np.uint8))
    img_ = img_.resize((out_size, out_size), Image.BICUBIC)
    return np.asarray(img_).astype(np.float32) / 255.0


def _gauss_kernel_1d_sigma(sigma: float, dtype=None, device=None) -> torch.Tensor:
    if sigma <= 0:
        k = torch.tensor([1.0], dtype=dtype, device=device)
        return k / k.sum()
    rad = int(math.ceil(3 * float(sigma)))
    x = torch.arange(-rad, rad + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(x**2) / (2 * (sigma**2)))
    return kernel / (kernel.sum() + 1e-12)


def gauss_blur_2d_tokens(tokens_bhwc: torch.Tensor, sigma: float) -> torch.Tensor:
    """Depth-wise separable blur in token space; keeps tensor shape unchanged."""
    if sigma is None or sigma <= 0:
        return tokens_bhwc

    B, H, W, C = tokens_bhwc.shape
    k1d = _gauss_kernel_1d_sigma(sigma, dtype=tokens_bhwc.dtype, device=tokens_bhwc.device)
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)
    pad_x = kx.shape[-1] // 2
    pad_y = ky.shape[-2] // 2

    x = tokens_bhwc.permute(0, 3, 1, 2)
    x_pad = F.pad(x, (pad_x, pad_x, 0, 0), mode="reflect")
    w_h = kx.repeat(C, 1, 1, 1)
    x = F.conv2d(x_pad, w_h, padding=0, groups=C)

    x_pad = F.pad(x, (0, 0, pad_y, pad_y), mode="reflect")
    w_v = ky.repeat(C, 1, 1, 1)
    x = F.conv2d(x_pad, w_v, padding=0, groups=C)

    return x.permute(0, 2, 3, 1)


def visualize_layers(
    image: np.ndarray,
    hidden_states: tuple[torch.Tensor, ...],
    model,
    layer_ids: list[int],
    H_tok: int,
    W_tok: int,
    patch: int,
    blur_sigma: float,
) -> plt.Figure:
    """Create the comparison figure containing the input image + token maps."""
    n_layers = len(layer_ids)
    n_cols = min(4, n_layers + 1)
    n_rows = math.ceil((n_layers + 1) / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axs = np.array(axs).reshape(-1)

    axs[0].imshow(image)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    for idx, layer_pick in enumerate(layer_ids, start=1):
        hs = hidden_states[layer_pick]
        _, total_tokens, C = hs.shape
        n_prefix = get_n_prefix_tokens(model, total_tokens, H_tok, W_tok)
        spatial = hs[:, n_prefix:, :]
        tokens_grid = spatial.view(1, H_tok, W_tok, C)
        tokens_grid = gauss_blur_2d_tokens(tokens_grid, sigma=blur_sigma)
        rgb = tokens_to_pca_rgb(tokens_grid[0])
        rgb = upsample_rgb(rgb, UPSAMPLE_VISUAL_TO)

        axs[idx].imshow(rgb)
        axs[idx].set_title(f"Layer {layer_pick} (patch={patch})\n{H_tok}×{W_tok} tokens")
        axs[idx].axis("off")

    for ax in axs[n_layers + 1 :]:
        ax.axis("off")

    fig.tight_layout()
    print(f"[DONE] Visualized layers: {layer_ids}")
    return fig


def visualize_inception_feature_map(
    image: np.ndarray, feature_map: torch.Tensor, blur_sigma: float
) -> plt.Figure:
    """Visualize an Inception feature map alongside the input tile."""
    feature_grid = feature_map.permute(0, 2, 3, 1)
    feature_grid = gauss_blur_2d_tokens(feature_grid, blur_sigma)
    rgb = tokens_to_pca_rgb(feature_grid[0])
    rgb = upsample_rgb(rgb, UPSAMPLE_VISUAL_TO)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(rgb)
    axs[1].set_title(f"Inception-BN feature map\n{rgb.shape[0]}×{rgb.shape[1]} pixels")
    axs[1].axis("off")

    fig.tight_layout()
    print("[DONE] Visualized Inception-BN feature map")
    return fig


def save_figure(fig: plt.Figure, descriptor: str, blur_sigma: float) -> Path:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{descriptor}_sigma{blur_sigma:.2f}.png"
    save_path = SAVE_DIR / fname
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def run_dino_pipeline(image: np.ndarray, blur_sigma: float) -> None:
    """Original ViT/DINO forward pass feeding token visualizations."""
    model_id = MODEL_REGISTRY[MODEL_CHOICE]
    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=LOCAL_ONLY)
    model = AutoModel.from_pretrained(
        model_id, local_files_only=LOCAL_ONLY, output_hidden_states=True
    ).to(DEVICE)
    model.eval()
    print(model)

    patch = resolve_patch_size(model)
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    preprocess = build_preprocess(mean, std)
    pixel_values = preprocess_image(image, preprocess).to(DEVICE)

    H_in, W_in = pixel_values.shape[-2:]
    if (H_in % patch) or (W_in % patch):
        print(
            f"[WARN] crop_to={CROP_TO} not divisible by patch_size={patch}. "
            f"Token grid will floor-divide to {(H_in // patch, W_in // patch)}."
        )
    H_tok, W_tok = H_in // patch, W_in // patch

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    hidden_states = outputs.hidden_states
    print(f"#layers: {len(hidden_states)}; first layer shape: {hidden_states[0].shape}")

    fig = visualize_layers(image, hidden_states, model, LAYER_LIST, H_tok, W_tok, patch, blur_sigma)
    print(
        f"Input tokens grid: {H_tok} x {W_tok}, patch={patch}, image={H_in}x{W_in}, blur_sigma={blur_sigma}"
    )
    descriptor = f"dino_offset{CROP_OFFSET}_patch{patch}_H{H_tok}_W{W_tok}"
    save_figure(fig, descriptor, blur_sigma)


def run_inception_pipeline(image: np.ndarray, blur_sigma: float) -> None:
    """Alternative CNN path using torchvision's Inception-BN pretrained weights."""
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights).to(DEVICE)
    model.eval()

    mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    std  = weights.meta.get("std",  [0.229, 0.224, 0.225])
    preprocess = build_preprocess(mean, std)

    pixel_values = preprocess_image(image, preprocess).to(DEVICE)

    captured = {}

    def _hook(_module, _inputs, output):
        captured["map"] = output.detach()

    handle = model.Mixed_7c.register_forward_hook(_hook)
    with torch.no_grad():
        _ = model(pixel_values)
    handle.remove()

    feature_map = captured.get("map")
    if feature_map is None:
        raise RuntimeError("Failed to capture feature map from Inception-BN.")

    fig = visualize_inception_feature_map(image, feature_map, blur_sigma)
    descriptor = f"inception_offset{CROP_OFFSET}_inception_bn_H{feature_map.shape[-2]}_W{feature_map.shape[-1]}"
    save_figure(fig, descriptor, blur_sigma)


def main() -> None:
    args = parse_args()
    image = load_histology_patch(IMAGE_PATH, CROP_OFFSET, CROP_SIZE)

    if args.backbone == "inception_bn":
        run_inception_pipeline(image, args.blur)
    else:
        run_dino_pipeline(image, args.blur)


if __name__ == "__main__":
    main()

# %%
