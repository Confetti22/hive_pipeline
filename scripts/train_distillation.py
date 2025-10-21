"""
# Old way (distillation.py):
# Edit the file to change settings, then:
python distillation.py

# New way (scripts/train_distillation.py):
# Edit config/distill.yaml, then:
python scripts/train_distillation.py -cfg config/distill.yaml

"""
import argparse
from pathlib import Path
from typing import List

import yaml
import torch
from torch.utils.data import DataLoader

from lib.distill import (
    GrayTiffDataset,
    Distiller,
)


def _validate_paths(paths: List[str]) -> List[str]:
    files: List[str] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            # collect tif/tiff files in the directory
            files.extend([str(f) for f in pp.rglob('*.tif')])
            files.extend([str(f) for f in pp.rglob('*.tiff')])
        elif pp.is_file():
            files.append(str(pp))
    return sorted(files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-cfg', required=True, help='Path to distillation YAML config')
    args = p.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    raw_paths = cfg.get('train_paths', [])
    train_paths = _validate_paths(raw_paths)
    if not train_paths:
        raise SystemExit("[ERR] No training images found. Provide train_paths (files or directories) in the config.")
    teacher_dir = cfg['teacher_dir']
    student_type = cfg.get('student_type', 'cnn')
    use_spatial_coords = bool(cfg.get('use_spatial_coords', False))
    batch_size = int(cfg.get('batch_size', 8))
    epochs = int(cfg.get('epochs', 50))
    lr = float(cfg.get('lr', 5e-4))
    wd = float(cfg.get('weight_decay', 0.05))
    num_workers = int(cfg.get('num_workers', 4))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = GrayTiffDataset(train_paths)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # Build student CNN when requested; users should provide a factory in their codebase
    def _build_student_cnn():
        from helper.model_basic import build_semantic_seg_model  # user-defined
        return build_semantic_seg_model()

    distiller = Distiller(
        teacher_dir=teacher_dir,
        student_type=student_type,
        student_cnn_builder=_build_student_cnn if student_type == 'cnn' else None,
        use_spatial_coords=use_spatial_coords,
    ).to(device)

    optim = torch.optim.AdamW(distiller.student.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith('cuda'))

    distiller.teacher.eval()
    distiller.student.train()

    for ep in range(epochs):
        for x_gray, image_ids in dl:
            x_gray = x_gray.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.startswith('cuda')):
                out = distiller(x_gray, image_ids)
                loss = out['loss']
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        print(f"[distill] epoch={ep+1:03d} loss={loss.item():.4f} L_feat={out['L_feat'].item():.4f} L_aff={out['L_aff'].item():.4f}")


if __name__ == '__main__':
    main()


