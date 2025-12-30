from __future__ import annotations
from typing import List

import torch
import torch.nn as nn


import torch

def count_model_size(model):
    "a general purpose model size counter, "
    "suitable for any model structure, can be used when torchsummay does not work"
    def _count_params(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def _state_dict_size_bytes(model):
        # includes parameters + buffers (e.g., norm stats), whatever is in state_dict
        sd = model.state_dict()
        return sum(t.numel() * t.element_size() for t in sd.values())

    total, trainable = _count_params(model)
    bytes_sd = _state_dict_size_bytes(model)

    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"FP32 param bytes: {total*4/1024**2:.2f} MiB  (approx, params only)")
    print(f"FP16 param bytes: {total*2/1024**2:.2f} MiB  (approx, params only)")
    print(f"State-dict bytes: {bytes_sd/1024**2:.2f} MiB  (current dtype in state_dict)")


class TeacherDinoV3(nn.Module):
    def __init__(self, model_dir: str, ckpt_path: str = None):
        super().__init__()
        from transformers import AutoModel, AutoConfig
            #only define the model with config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if ckpt_path and ckpt_path.endswith('.pth'):
            config = AutoConfig.from_pretrained(model_dir)
            self.vit= AutoModel.from_config(config)
            ckpt = torch.load(ckpt_path, map_location=device)
            # print(f"ckpt keys: {ckpt.keys()}")
            # print(f"model_state_dict keys: {ckpt['model_state_dict'].keys()}")
            # Remove 'vit.' prefix from keys and filter out projection head keys before loading
            state_dict = ckpt['model_state_dict']
            filtered_state_dict = {}
            for k, v in state_dict.items():
                # Skip projection head keys
                if k.startswith('projection_head.'):
                    continue
                # Remove 'vit.' prefix if present
                if k.startswith('vit.'):
                    filtered_state_dict[k.replace('vit.', '')] = v
                else:
                    filtered_state_dict[k] = v
            res = self.vit.load_state_dict(filtered_state_dict)
            print(f"load_state_dict result: {res}")
        else:
            #define model and load weights from model_dir
            self.vit = AutoModel.from_pretrained(
                model_dir, local_files_only=True, output_hidden_states=True
            )
        self.embed_dim = self.vit.config.hidden_size
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward_tokens(self, x_rgb: torch.Tensor, tap_blocks_0based: List[int]):
        out = self.vit(x_rgb, output_hidden_states=True)
        hs = out.hidden_states
        tokens_list = []
        for blk in tap_blocks_0based:
            h = hs[blk + 1]
            tokens_list.append(h[:, 5:, :])
        return tokens_list

