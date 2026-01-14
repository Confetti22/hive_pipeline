#~~~~~~~ weighted l1 loss ~~~~~~~~#
from lib.loss.ce_dice_combo import ComboLoss
from lib.utils.loss_utils import compute_class_weights_from_dataset
from lib.arch.segmodel import Modelsegmodel
import torch
from torch.utils.data import DataLoader, Dataset

def train_seghead(segmodel: Modelsegmodel,
                  dataset: Dataset,
                  n_classes: int,
                  device: str = "cuda",
                  epochs: int = 5,
                  batch_size: int = 64,
                  precomute_feat: bool = False,
                  lr: float = 1e-3) -> None:
    """Train only the seghead with the backbone frozen.

    Args:
        segmodel: Modelsegmodel (backbone frozen)
        dataset: training dataset from sparse labels
        n_classes: number of classes
        device: 'cuda' or 'cpu'
        epochs: small number for interactivity
        batch_size: mini-batch size
        lr: learning rate
    """
    
    # Freeze backbone should be done in model initialization
    segmodel.seg_model.to(device)

    #drop_last False to ensure nonempty loader when  len(ds) ==1 (this is true when input img and train_roi is the same) 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last= False)
    opt = torch.optim.AdamW((p for p in segmodel.seg_model.parameters() if p.requires_grad), lr=lr)
    

    class_weights = compute_class_weights_from_dataset(dataset, num_classes=n_classes,recon_target_flag=False)
    loss_fn = ComboLoss(class_weights=class_weights, focal=True)

    # before training loop (once)
    torch.autograd.set_detect_anomaly(True)
    for n_epoch in range(epochs):
        for x, y in loader:
            if not precomute_feat:
                x = x.to(device)

            y = y.to(device)
            y = y.squeeze(1) 
            
            if precomute_feat:  #seg_model is just the seghead
                x = [feat.squeeze(0)for feat in x] #dataloader will add extra batch dim
                logits = segmodel.seg_model(x) 
            else:
                if x.shape[2] ==1:
                    x = x.squeeze(2)
                if segmodel.name == "DPT" and x.dim() == 5:
                    # slice 3D into 2D batches externally
                    B, C, D, H, W = x.shape
                    x2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
                    logits2d = segmodel.seg_model(x2d)  # [B*D, C', H, W]
                    logits = logits2d.reshape(B, D, n_classes, H, W).permute(0, 2, 1, 3, 4)
                else:
                    logits = segmodel.seg_model(x) #[B,C,D,H,W] or [B,C,H,W]

            # Align logits & labels to same spatial dims
            if logits.dim() == y.dim() + 1:# 2D: logits [B,C,H,W], y [B,H,W] OK# 3D: logits [B,C,D,H,W], y [B,D,H,W] OK
                pass
            else:
                raise RuntimeError("Unexpected logits/labels dims mismatch")
            
            logits_flat = logits.permute(0,*range(2, logits.ndim),1)[ y>= 0]  # [N, K]
            labels_flat = y[ y>= 0]

            # inside the loop, right before backward
            logits_flat = logits_flat.contiguous()
            assert logits_flat.dtype in (torch.float32, torch.float16, torch.bfloat16)
            assert logits_flat.requires_grad, "logits_flat must require grad"
            assert labels_flat.dtype == torch.long, f"labels dtype is {labels_flat.dtype}, expected long"
            assert labels_flat.min().item() >= 0 and labels_flat.max().item() < n_classes, \
                f"label out of range: [{labels_flat.min().item()}, {labels_flat.max().item()}]"
            
            total_loss, ce_loss, dice_loss= loss_fn(logits_flat, labels_flat.long())

            opt.zero_grad(set_to_none=True)
            
            total_loss.backward()  # if it fails, detect_anomaly will print the offending op
            opt.step()

            print(f"training: epoch:{n_epoch}:train_loss: {total_loss.item():.4f}  "
                    f"(ce={ce_loss.item():.4f}, dice={dice_loss.item():.4f})")


