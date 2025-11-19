#%%
import numpy as np
def _make_blend_weight_2d(h: int, w: int) -> np.ndarray:
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    dist = np.sqrt(y * y + x * x)
    if dist.max() > 0:
        dist /= dist.max()
    weight = 1.0 - dist
    return weight.astype(np.float32)

print(_make_blend_weight_2d(5,5))

# %%
