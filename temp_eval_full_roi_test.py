import numpy as np
import torch
from interactive_svc_single_viewer import Modelsegmodel, eval_full_roi

class DummySeg(torch.nn.Module):
    def __init__(self, n_classes: int, feat_dim: int = 4):
        super().__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self._feature_map = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1)
        xx = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w)
        logits = torch.cat([
            yy.expand(b, 1, h, w),
            xx.expand(b, 1, h, w),
            torch.zeros(b, self.n_classes - 2, h, w, device=x.device)
        ], dim=1)

        feat = torch.stack([
            yy.squeeze(0).expand(h, w),
            xx.squeeze(0).expand(h, w),
            torch.zeros(h, w),
            torch.ones(h, w)
        ], dim=-1)
        self._feature_map = feat.cpu().numpy().astype(np.float32)
        return logits

    def get_feature_map(self):
        return self._feature_map

image = np.random.randint(0, 65535, size=(512, 1536), dtype=np.uint16)
segmodel = Modelsegmodel(name="dummy", dims=2, seg_model=DummySeg(n_classes=3), n_classes=3)

pred, feat = eval_full_roi(
    segmodel,
    image,
    device="cpu",
    tile=(512, 512),
    capture_features=True,
    tv_denoise_weight=0.0,
    overlap=0.25,
)

print("Prediction shape:", pred.shape)
print("Feature shape:", None if feat is None else feat.shape)
