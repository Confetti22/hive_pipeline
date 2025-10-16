# ===========================
# v2 vs v1 — key differences
# ===========================
# [NEW] Deep-learning stack: torch, torchvision, model loading (AE/MLP/Inception) and a shared model_dict.
# [NEW] Project path bootstrap (sys.path insert) so imports like lib.arch.* work when running from /test.
# [NEW] Async rendering for napari via NAPARI_ASYNC=1.
# [CHANGED] SimpleViewer -> SimpleViewer2 (extended viewer widget).
# [NEW] OneDimStatis_roi dock widget for ROI statistics / model inference on selections.
# [NEW] Extra key binding on viewer1 to toggle 'segout' layer visibility (v1 only toggled on viewer2).
# [SAME] MultipleViewerWidget usage, two-viewer layout, geometry handling, Seger widget, and 'v' key on viewer2
# [REMOVED in v2] The commented magicgui Z-slider stub from v1 isn’t present here.
# ===========================

import napari
import torch  # [NEW] Deep learning
import numpy as np
from magicgui import magicgui
from helper.simple_viewer import SimpleViewer2  # [CHANGED] v1 used SimpleViewer
from helper.image_seger import SimpleSeger2
from helper.napari_view_utilis import MultipleViewerWidget, toggle_layer_visibility
from helper.one_dim_statis import OneDimStatis_roi  # [NEW] ROI stats widget

# ----- Runtime/project setup -----
import sys  # [NEW]
import os   # [NEW]

# [NEW] Ensure repo root is importable when running from subfolders (e.g., tests/scripts)
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

# [NEW] Enable napari async rendering (better responsiveness with model calls / I/O)
os.environ["NAPARI_ASYNC"] = "1"

# [NEW] Model/config imports
from lib.arch.ae_old import (
    build_final_model,
    load_compose_encoder_dict,
    build_encoder_model,
    load_encoder2encoder,
)
from config.load_config import load_cfg
import torchvision.models as models
from torchvision.models import Inception_V3_Weights

# ----- Model/config init (NEW in v2) -----
device = 'cuda'

# [NEW] Load pipeline config and set encoder/MLP specifics
args = load_cfg('config/t11_3d.yaml')
args.last_encoder = True

#if avgpooling is not added, the feature_map is noisy
args.avg_pool_size = (8, 8, 8)
# args.avg_pool_size = None 

# the second row comments is for later contrastive learning result
args.mlp_filters = [96, 48, 24, 12]
# args.mlp_filters = [96, 64, 32, 12]

# [NEW] Composite model (AE + MLP)
cmpsd_model = build_final_model(args)
cmpsd_model.eval().to(device)
cnn_ckpt_pth = '/home/confetti/data/weights/t11_3d_ae_best2.pth'
mlp_ckpt_pth = '/home/confetti/data/weights/t11_3d_mlp_best_new_format.pth'
# mlp_ckpt_pth = '/home/confetti/e5_workspace/hive1/outs/contrastive_run_t1779/test_on_rhems_numparis16384_batch4096_nview4_d_near8_shuffle20_csine_anllr_/checkpoints/epoch_8700.pth'

mlp_weights_dict = torch.load(mlp_ckpt_pth)
# mlp_weights_dict = torch.load(mlp_ckpt_pth)['model']
load_compose_encoder_dict(cmpsd_model, cnn_ckpt_pth, mlp_weight_dict=mlp_weights_dict, dims=args.dims)

# [NEW] Standalone encoder model (AE branch only)
encoder_model = build_encoder_model(args, dims=3)
encoder_model.eval().to(device)
load_encoder2encoder(encoder_model, cnn_ckpt_pth)

# [NEW] Pretrained backbone for auxiliary features/embeddings
weights = Inception_V3_Weights.DEFAULT
incep_model = models.inception_v3(weights=weights, progress=True)
incep_model.eval().to(device)

# [NEW] Shared model registry for UI widgets/tools to consume
model_dict = {
    'mlp': cmpsd_model,
    'ae': encoder_model,
    'inception': incep_model,
}

# ----- Napari UI (mostly same as v1) -----

# main viewer (same as v1)
viewer1 = napari.Viewer(ndisplay=2)

# sub viewer container (same as v1)
dock_widget = MultipleViewerWidget(viewer1)
viewer1.window.add_dock_widget(dock_widget, name="multiViewer")

# keep window geometry consistent (same as v1)
origin_geo = viewer1.window.geometry()
viewer1.window.set_geometry(origin_geo[0], origin_geo[1], origin_geo[2], origin_geo[3])

# sub viewer access (same as v1)
viewer2 = viewer1.window._dock_widgets['multiViewer'].widget().viewer_model1

# Viewer widgets
# [CHANGED] Use SimpleViewer2 (v1 used SimpleViewer)
simple_viewer_widget = SimpleViewer2(viewer1, viewer2)

# same Seger widget as v1
simple_seger_widegt = SimpleSeger2(viewer1, viewer2, simple_viewer_widget)

# [NEW] ROI statistics widget that can use the loaded models via model_dict
statis_widget = OneDimStatis_roi(
    viewer=viewer1,
    image_layer=simple_viewer_widget.roi_layer,
    model_dict=model_dict,
)

# Key bindings
# same binding on viewer2 as v1 (toggle masks/regions/polygons)
@viewer2.bind_key('v')
def toggle_mask_viewer2(_module):
    print(f"press 'v' at  {_module}")
    toggle_layer_visibility(layers=viewer2.layers, name_patterns=['mask', 'region', 'polygon'])

# [NEW] Additional binding on viewer1 to toggle predicted segmentation output layer(s)
@viewer1.bind_key('v')
def toggle_segout_viewer1(_module):
    print(f"press 'v' at  {_module}")
    toggle_layer_visibility(layers=viewer2.layers, name_patterns=['segout'])

# Dock widgets layout (right/left/left), same order plus the new stats pane
viewer1.window.add_dock_widget(simple_viewer_widget, area='right')
viewer1.window.add_dock_widget(simple_seger_widegt, area='left')
viewer1.window.add_dock_widget(statis_widget, area='left')  # [NEW]

napari.run()