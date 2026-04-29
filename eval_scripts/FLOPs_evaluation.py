# pip install thop
import torch
from thop import profile, clever_format
import torch
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
import argparse
import yaml
import torch
from lib.distill import (
    GrayTiffDataset,
    Distiller,
)

from lib.distill.student import build_student_cnn


def main():

    p = argparse.ArgumentParser()
    p.add_argument('-cfg', default='config/distill.yaml', help='Path to distillation YAML config')
    args = p.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    max_train_samples = cfg.get('max_train_samples', None)
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
    teacher_dir = cfg['teacher_dir']
    ckpt_path = cfg.get('ckpt_path', None)
    student_type = cfg.get('student_type', 'cnn')
    tinyvit_input_type = cfg.get('tinyvit_input_type', 'rgb')
    tinyvit_implementation = cfg.get('tinyvit_implementation', 'local')
    tinyvit_depth = int(cfg.get('tinyvit_depth', 12))
    tinyvit_embed_dim = int(cfg.get('tinyvit_embed_dim', 192))
    tinyvit_num_heads = cfg.get('tinyvit_num_heads', None)
    if tinyvit_num_heads is not None:
        tinyvit_num_heads = int(tinyvit_num_heads)
    tinyvit_mlp_ratio = float(cfg.get('tinyvit_mlp_ratio', 4.0))
    teacher_feature_layers = cfg.get('teacher_feature_layers', [2, 6, 11])
    student_feature_layers = cfg.get('student_feature_layers', [2, 6, 11])
    lambda_feat = float(cfg.get('lambda_feat', 1.0))
    lambda_aff = float(cfg.get('lambda_aff', 0.25))
    use_spatial_coords = bool(cfg.get('use_spatial_coords', False))
    feature_loss_type = cfg.get('feature_loss_type', 'cosine')  # 'cosine' or 'mse'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    distiller = Distiller(
        teacher_dir=teacher_dir,
        ckpt_path=ckpt_path,
        student_type=student_type,
        student_cnn_builder= build_student_cnn(model_type='simple') if student_type == 'cnn' else None,
        tinyvit_input_type=tinyvit_input_type,
        tinyvit_implementation=tinyvit_implementation,
        tinyvit_depth=tinyvit_depth,
        tinyvit_embed_dim=tinyvit_embed_dim,
        tinyvit_num_heads=tinyvit_num_heads,
        tinyvit_mlp_ratio=tinyvit_mlp_ratio,
        teacher_feature_layers=teacher_feature_layers,
        student_feature_layers=student_feature_layers,
        lambda_feat=lambda_feat,
        lambda_aff=lambda_aff,
        use_spatial_coords=use_spatial_coords,
        feature_loss_type=feature_loss_type,
    ).to(device)

    # 2. 创建一个符合您真实推理尺寸的 Dummy Tensor (B, C, H, W)
    dummy_input_teacher = torch.randn(1, 3, 512, 512).to(device)
    if tinyvit_input_type.lower() in ["gray", "grayscale", "single"]:
        dummy_input_student = torch.randn(1, 1, 512, 512).to(device)
    else:
        dummy_input_student = torch.randn(1, 3, 512, 512).to(device)

    print(f"{dummy_input_student.shape}")
    exit(0)
    # 3. 计算教师模型 (注意：只计算特征提取的前向传播过程)
    macs_teacher, params_teacher = profile(distiller.teacher.vit, inputs=(dummy_input_teacher, ))
    flops_teacher = macs_teacher * 2
    flops_teacher, params_teacher = clever_format([flops_teacher, params_teacher], "%.2f")
    print(f"Teacher - FLOPs: {flops_teacher}, Params: {params_teacher}")

    # 4. 计算学生模型
    macs_student, params_student = profile(distiller.student.vit, inputs=(dummy_input_student, ))
    flops_student = macs_student * 2
    flops_student, params_student = clever_format([flops_student, params_student], "%.2f")
    print(f"Student - FLOPs: {flops_student}, Params: {params_student}")

if __name__ == '__main__':
    main()
