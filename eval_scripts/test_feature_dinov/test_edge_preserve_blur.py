
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



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHOICE = "dinov3_vits16"  # {'dinov2_vitl14', 'dinov3_vits16'}
MODEL_REGISTRY = {
    "dinov2_vitl14": "facebook/dinov2-large",  # ViT-L/14 (patch=14)
    "dinov3_vits16": "/home/confetti/e5_workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m",
}
LOCAL_ONLY = MODEL_CHOICE == "dinov3_vits16"

