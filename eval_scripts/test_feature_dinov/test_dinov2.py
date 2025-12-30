#%%
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

#######       prepare image     ########
import tifffile as tif
from skimage import io
img = io.imread("/home/confetti/data/mousebrainatlas/histology/102117913_d0.png")

print(f"{img.shape= }")
print(f"{img.dtype= }")
import matplotlib.pyplot as plt
offset = (1800,1800)
size = 1000 
img = img[offset[0]:offset[0]+size, offset[1]:offset[1]+size, :]
plt.imshow(img)
#%%

input_size = 980 
import torchvision.transforms as T
preprocess = T.Compose([
    T.ToTensor(),
    T.Resize(input_size), T.CenterCrop(input_size),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

#!/usr/bin/env python3
import torch
from PIL import Image
from functools import partial
from collections import defaultdict
from transformers import AutoModel, AutoImageProcessor
from torchvision import transforms
from transformers import  Dinov2Model
import math
import numpy as np
# ---------- 你的 hook 容器 ----------
FEATURE_STORE = defaultdict(list)      # {layer_name: [Tensor, ...]}
HOOK_HANDLES  = []                     # for clean removal
LAYER_ORDER   = []

def _hook(layer_name, module, inp, out):
    """
    对 ViT 的 Block 来说:
      out: [B, 1+HW, C] — 含 CLS(第0位) 与 patch tokens
    我们放到 CPU，避免占 GPU 显存。
    """
    FEATURE_STORE[layer_name].append(out.detach().cpu())

def clear_hooks():
    for h in HOOK_HANDLES:
        h.remove()
    HOOK_HANDLES.clear()
    FEATURE_STORE.clear()
    LAYER_ORDER.clear()

def register_hook_on_name(model, target_full_name: str):
    """
    在任意模块（包括非叶子模块）上挂 hook。
    target_full_name 例如: "blocks.11" 或 "blocks.17"
    """
    for full_name, m in model.named_modules():
        if full_name == target_full_name:
            LAYER_ORDER.append(full_name)
            HOOK_HANDLES.append(m.register_forward_hook(partial(_hook, full_name)))
            return True
    return False

# ---------- 帮助函数 ----------
def infer_grid_hw_from_tokens(num_tokens: int, has_cls: bool = True):
    """
    从 token 数量反推 (H', W')。有 CLS 时 num_tokens = 1 + H'*W'
    """
    if has_cls:
        num_tokens -= 1
    hw = int(math.isqrt(num_tokens))
    assert hw * hw == num_tokens, f"Token数 {num_tokens} 不是正方形网格！"
    return hw, hw

def tokens_to_grid(tokens: torch.Tensor, has_cls: bool = True):
    """
    tokens: [B, 1+HW, C] 或 [B, HW, C]
    返回: (cls, grid) 其中
      cls  : [B, C]
      grid : [B, H', W', C]
    """
    if has_cls:
        cls = tokens[:, 0]                        # [B, C]
        patch = tokens[:, 1:]                     # [B, HW, C]
    else:
        cls = None
        patch = tokens                            # [B, HW, C]
    B, HW, C = patch.shape
    H, W = infer_grid_hw_from_tokens(HW, has_cls=False)
    grid = patch.view(B, H, W, C)                 # [B, H', W', C]
    return cls, grid


#%%
# ---------- 准备模型 & 预处理 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

# 从 torch.hub 拉取 DINOv2 ViT-L/14（会返回带 head 的模型；我们只做前向抽 token）
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model.eval().to(device)



# Map friendly names -> your local dir OR HF id (change paths if you have local copies)
MODEL_REGISTRY = {
    # If you have local folders, put them here instead of the HF ids.
    'dinov2_vitl14': "facebook/dinov2-large",   # ViT-L/14 (patch=14)
    'dinov3_vits16': "/home/confetti/e5_workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m",  # ViT-S/16 (patch=16)
}

model_choice = 'dinov3_vits16'
model_id_or_dir = MODEL_REGISTRY[model_choice]
processor = AutoImageProcessor.from_pretrained(model_id_or_dir, local_files_only= False)
model = AutoModel.from_pretrained(
    model_id_or_dir, local_files_only=False, output_hidden_states=True
).to(device).eval()


from torchsummary import summary
print(model)
# summary(model, (3, 224, 224))
#%%

# ---------- 选择中间层并注册 hook ----------
# ViT-L/14 有 24 个 blocks，索引 0..23；中位数可取 11 或 12
middle_block_idx = 7
block_name = f"layer.{middle_block_idx}"
ok = register_hook_on_name(model, block_name)
assert ok, f"没有找到模块 {block_name}，请检查命名。"



# ---------- 前向，抽取 token ----------
with torch.no_grad():
    inputs = preprocess(img).unsqueeze(0).to(device)
    _ = model(inputs)  # 正常前向一次即可触发 hook


# 取回该层输出 token
tokens = FEATURE_STORE[block_name][0]      # [B, 1+HW, C] （在 CPU 上）
print("hooked tokens:", tokens.shape)

# 拆出 CLS 与 patch 网格
cls_tok, grid_tok = tokens_to_grid(tokens, has_cls=True)  # cls:[B,C], grid:[B,H',W',C]
print("tokesns", tokens.shape,"cls:", cls_tok.shape, "grid:", grid_tok.shape)
clear_hooks()


np_grid_tok = grid_tok.detach().cpu().numpy()   # [H', W', C] -> [H'*W', C]
feats_lst = np.reshape(np_grid_tok, (-1, np_grid_tok.shape[-1]))
print(f"{feats_lst.shape= }")

from confettii.plot_helper import three_pca_as_rgb_image
rgb_img = three_pca_as_rgb_image(feats_lst, final_image_shape=grid_tok.shape[1:3])
import matplotlib.pyplot as plt


# Visualize input, PCA(grid_tok), PCA(grid_tok2)
figs, axs = plt.subplots(1,3, figsize=(24,10))
axs[0].imshow(img)
axs[0].set_title("Input Image")
axs[1].imshow(rgb_img)
axs[1].set_title("PCA of Features (hooked model)")
# %%

  
#%%
# %%
