from dataclasses import dataclass
from typing import Dict, Optional
from torch import nn
import torch
import torch.nn.functional as F

##### DPT ######
from transformers import AutoModel
from lib.arch.segdino import Dinov3HFBackbone, DPT, DPTHead_warped, LinearTokenSeg
######inception_v3 #######
from lib.arch.inception import InceptionBackbone,Inception_V3_Weights,InceptionSegHead,InceptionSegModel, InceptionLinearSegModel
from lib.arch.ae import ConvMLP
from lib.arch.seg import SimpleSegmodel

@dataclass
class Modelsegmodel:
    name: str
    dims: int
    seg_model: nn.Module|DPT| LinearTokenSeg|InceptionSegModel|InceptionLinearSegModel| SimpleSegmodel
    n_classes: int


def build_old_cnn_seg(dims: int, n_classes: int, model_dir = None,linear_prob: bool = True,lock_backbone=True) -> Modelsegmodel:
    """Build cmpsd backbone + ConvSegHead.

    Args:
        dims: 2 or 3
        n_classes: number of output classes (incl. background)
        feat_channels: channels produced by backbone
    Returns:
        Modelsegmodel
    """


    #todo: try different mlp weights at different epoch: 50, 200, 2000
    t1779 = True
    if t1779:
        from lib.arch.ae_old import build_contrastive_model
        args = load_cfg('config/t11_3d.yaml')
        args.last_encoder = True

        #if avgpooling is not added, the feature_map is noisy
        args.avg_pool_size = (8, 8, 8)
        # args.avg_pool_size = None 

        # the second row comments is for later contrastive learning result
        args.mlp_filters = [96, 48, 24, 12]
        # args.mlp_filters = [96, 64, 32, 12]

        # [NEW] Composite model (AE + MLP)
        cmpsd_model = build_contrastive_model(args)
        cmpsd_model.eval().to
        cnn_ckpt_pth = '/home/confetti/data/weights/t11_3d_ae_best2.pth'
        mlp_ckpt_pth = '/home/confetti/data/weights/t11_3d_mlp_best_new_format.pth'
        # mlp_ckpt_pth = '/home/confetti/e5_workspace/hive1/outs/contrastive_run_t1779/test_on_rhems_numparis16384_batch4096_nview4_d_near8_shuffle20_csine_anllr_/checkpoints/epoch_8700.pth'

        mlp_weights_dict = torch.load(mlp_ckpt_pth)
        # mlp_weights_dict = torch.load(mlp_ckpt_pth)['model']
        load_compose_encoder_dict(cmpsd_model, cnn_ckpt_pth, mlp_weight_dict=mlp_weights_dict, dims=args.dims)


    else:
        from config.load_config import load_cfg
        cfg = load_cfg('config/seghead.yaml')

        cfg.filters = [32,64]
        cfg.last_encoder= len(cfg.filters)==3

        cfg.mlp_filters =[64,32,24,12]
        cfg.feats_level = len(cfg.filters)
        cfg.feats_avg_kernel = 8

        cfg.dims = dims

        from lib.arch.ae_old import build_cmpsd_model,load_compose_encoder_dict
        cmpsd_model = build_cmpsd_model(cfg)
        cmpsd_model.eval()

        if lock_backbone:
            for param in cmpsd_model.parameters():
                param.requires_grad = False


        # cnn_ckpt_pth = "/share/home/shiqiz/data/weights/ae_feats_nissel_v1_roi1_decaylr_e1600.pth"
        # mlp_ckpt_pth = "/share/home/shiqiz/workspace/hive1/outs/contrastive_run_rm009/rm009_whole_brain/postopk_8_numparis16384_batch4096_nview4_d_near4_shuffle50/model_epoch_100.pth"
        
        cnn_ckpt_pth = "/home/confetti/data/weights/ae_feats_nissel_v1_roi1_decaylr_e1600.pth"
        mlp_ckpt_pth = "/home/confetti/e5_workspace/hive1/outs/contrastive_run_rm009/rm009_whole_brain/postopk_8_numparis16384_batch4096_nview4_d_near4_shuffle50/model_epoch_100.pth"

        mlp_ckpt = torch.load(mlp_ckpt_pth)
        load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_weight_dict=mlp_ckpt,dims=dims) #this pretrained model is 3d
        print(f"load cnn from {cnn_ckpt_pth} and mlp from {mlp_ckpt_pth}")


    from lib.arch.ae import ConvMLP
    from lib.arch.seg import SimpleSegmodel
    
    print(f"{linear_prob= }")
    if linear_prob:
        seg_head = ConvMLP(filters=[cfg.mlp_filters[-1], n_classes], l2_norm=False, last_act=False, dims=dims).train()
    else:
        seg_head = ConvMLP(filters=[cfg.mlp_filters[-1],cfg.mlp_filters[-1],n_classes],l2_norm=False,last_act=False,dims=dims).train() 
    
    seg_model = SimpleSegmodel(cmpsd_model,seg_head)

    return Modelsegmodel("cnn_seg_old", dims ,seg_model,n_classes)


def build_cnn_seg(dims: int, n_classes: int, model_dir = None,linear_prob: bool = False,lock_backbone=True) -> Modelsegmodel:
    """Build cmpsd backbone + ConvSegHead.

    Args:
        dims: 2 or 3
        n_classes: number of output classes (incl. background)
        feat_channels: channels produced by backbone
    Returns:
        Modelsegmodel
    """
    from config.load_config import load_cfg
    cfg = load_cfg('/home/confetti/e5_workspace/hive1_pipeline/runs/contrastive/onestage_batch2028_nview2_infolossFalse_t1779_2um/config.yaml')
    avg_pool_size = 4
    cfg.dims = dims
    cfg.avg_pool_size = [avg_pool_size]*dims



    #todo: try different mlp weights at different epoch: 50, 200, 2000
    from lib.arch.ae import build_contrastive_model,load_compose_encoder_dict
    contrast_model = build_contrastive_model(cfg)
    contrast_model.eval()
    contrast_model.off_proj() #discard projecitonhead in downstream task

    if lock_backbone:
        for param in contrast_model.parameters():
            param.requires_grad = False

    if model_dir:
        if isinstance(model_dir, str):
            cmpsd_ckpt = torch.load(model_dir)
            contrast_model.load_state_dict(cmpsd_ckpt)
            print(f"load weight at {model_dir}")
        elif isinstance(model_dir, dict):
            cnn_ckpt_pth = model_dir['cnn']
            mlp_ckpt_pth = model_dir['mlp']
            mlp_ckpt = torch.load(mlp_ckpt_pth)
            load_compose_encoder_dict(contrast_model,cnn_ckpt_pth,mlp_weight_dict=mlp_ckpt,dims=dims) #this pretrained model is 3d
            
            # cnn_ckpt_pth = "/home/confetti/data/weights/ae_feats_nissel_v1_roi1_decaylr_e1600.pth"
            # mlp_ckpt_pth = "/home/confetti/e5_workspace/hive1/outs/contrastive_run_rm009/ae_mlp_rm009_v1/FEATl2_avg8_LOSSpostopk_numparis16384_batch4096_nview4_d_near6_shuffle20_cosdecay_valide_with_avgpool/checkpoints/epoch_4000.pth"
            # mlp_ckpt = torch.load(mlp_ckpt_pth)['model']
            # load_compose_encoder_dict(contrast_model,cnn_ckpt_pth,mlp_weight_dict=mlp_ckpt,dims=dims) #this pretrained model is 3d
        else:
            print("model_dir format error, should be str or dict")
            return 
    else:
        print(f"model initialzed with random weights")

    
    if linear_prob:
        seg_head = ConvMLP(filters=[cfg.mlp_filters[-1], n_classes], l2_norm=False, last_act=False, dims=dims).train()
    else:
        seg_head = ConvMLP(filters=[cfg.mlp_filters[-1],cfg.mlp_filters[-1],n_classes],l2_norm=False,last_act=False,dims=dims).train() 
    
    seg_model = SimpleSegmodel(contrast_model,seg_head)

    return Modelsegmodel("cnn_seg", dims ,seg_model,n_classes)



def build_dpt(dims: int, n_classes: int, model_dir: Optional[str] = None, linear_prob: bool = False, smooth_params=(16,4,1)) -> Modelsegmodel:
    """Build DPT with DINOv3-like backbone + DPTHead.

    Notes:
        - For dims=3, we evaluate per-slice; backbone remains 2D.
        - seghead expects 2D features; for 3D ROI we loop slices externally.
    """
    if model_dir is None:
        model_dir = "/home/confetti/e5_workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m"
    print(f"build_dpt with {model_dir}")
    hf_backbone = AutoModel.from_pretrained(
        model_dir, local_files_only=True, output_hidden_states=True
    ).eval()
    backbone = Dinov3HFBackbone(hf_backbone)
    
    if linear_prob:
        seg_model = LinearTokenSeg(backbone, nclass=n_classes)
    else:
        seg_model = DPT(nclass=n_classes,backbone=backbone,smooth_params= smooth_params)
    
    seg_model.train()
    seg_model.lock_backbone()

    print("\n","unfrozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if  p.requires_grad],"\n")

    return Modelsegmodel("DPT", dims,seg_model,n_classes)


def build_seg_head(dims: int, n_classes: int,patch_h, patch_w) -> Modelsegmodel:

    seg_model = DPTHead_warped(n_classes, 768, features=128,use_bn=False , out_channels=[96, 192, 384, 768],patch_h=patch_h,patch_w=patch_w)
    seg_model.train()

    print("\n","unfrozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if  p.requires_grad],"\n")

    return Modelsegmodel("DPT", dims,seg_model,n_classes)




from lib.distill.student import TinyVitBackbone, TinyViTWithTaps, TinyViTWithTapsTimm
def build_tinyvit_dpt(dims:int, n_classes:int, model_dir: Optional[str] = None, linear_prob: bool = False,lock_backbone=True) -> Modelsegmodel:

    
    _model = TinyViTWithTaps( embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0).eval()
    # _model.register_taps([2,5,8,11])

    if model_dir :
        print(f"build_tinyvit_dpt with {model_dir}")
        state_dict = torch.load(model_dir)
        result=_model.load_state_dict(state_dict)
        print(f"load tinyvit model state dict result: {result}")
    else:
        print(f"build_tinyvit_dpt with random initialized weights")

    backbone  = TinyVitBackbone(model=_model) 

    if linear_prob:
        seg_model = LinearTokenSeg(backbone, nclass=n_classes)
    else:
        seg_model = DPT(nclass=n_classes,backbone=backbone)
    
    seg_model.train()
    if lock_backbone:
        seg_model.lock_backbone()

    print("\n","unfrozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if  p.requires_grad],"\n")
    return Modelsegmodel("s_tinyvit", dims,seg_model,n_classes)





def build_tinyvittimm_dpt(dims:int, n_classes:int, model_dir: Optional[str] = None, linear_prob: bool = False) -> Modelsegmodel:

    if model_dir is None:
        model_dir = 'runs/_distill_cnn_test10_tinyvittimm_mullayer_nomixup_cosine_proj_t1779/student_epoch_070.pth'
    
    _model = TinyViTWithTapsTimm().eval()
    # _model.register_taps([2,5,8,11])

    state_dict = torch.load(model_dir)
    result=_model.load_state_dict(state_dict)
    print(f"build_tinyvit_timm with {model_dir}")
    print(f"load tinyvittimm model state dict result: {result}")

    backbone  = TinyVitBackbone(model=_model) 

    if linear_prob:
        seg_model = LinearTokenSeg(backbone, nclass=n_classes)
    else:
        seg_model = DPT(nclass=n_classes,backbone=backbone)
    
    seg_model.train()
    seg_model.lock_backbone()

    print("\n","unfrozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if  p.requires_grad],"\n")
    return Modelsegmodel("s_tinyvittimm", dims,seg_model,n_classes)
  

def build_and_load_weights_dpt(dims: int, ) -> Modelsegmodel:
    """Build DPT with DINOv3-like backbone + DPTHead.

    Notes:
        - For dims=3, we evaluate per-slice; backbone remains 2D.
        - seghead expects 2D features; for 3D ROI we loop slices externally.
    """

    n_classes = 9  #predefined number of classes for this pretrained model
    model_dir = "/home/confetti/e5_workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m"# ViT-S/16 (patch=16)
    hf_backbone = AutoModel.from_pretrained(
        model_dir, local_files_only=True, output_hidden_states=True
    )
    backbone = Dinov3HFBackbone(hf_backbone)
    seg_model = DPT(nclass=n_classes,backbone=backbone)
    # ckpt= torch.load("/home/confetti/e5_workspace/hive1/outs/seg_dino/seg_dino_1zmip/model_epoch_3.pth")
    ckpt= torch.load("/home/confetti/e5_workspace/hive1/outs/seg_dino/seg_dino_nomask_with_layer2_5_8_11_metrics_batch16_1zmip/model_epoch_30.pth")
    weights = ckpt['seg_model']

    result = seg_model.load_state_dict(weights)
    print(result)

    #freeze backbone
    seg_model.lock_backbone()
    seg_model.eval()

    print("\n","unfrozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if  p.requires_grad],"\n")

    return Modelsegmodel("DPT", dims,seg_model,n_classes)



def build_inception_v3(dims: int, n_classes: int, linear_prob: bool = False) -> Modelsegmodel:
    """Build inception_v3 backbone + lightweight multi-scale seg head."""
    if dims != 2:
        raise ValueError("inception_v3 model currently supports 2D inputs only.")

    try:
        backbone = InceptionBackbone(Inception_V3_Weights.IMAGENET1K_V1)
    except Exception as exc:
        print(f"Falling back to randomly initialized InceptionV3 weights due to: {exc}")
        backbone = InceptionBackbone(weights=None)

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    if linear_prob:

        seg_model = InceptionLinearSegModel(backbone, n_classes)
    else:
        head = InceptionSegHead(backbone.out_channels, n_classes=n_classes, proj_dim=128, fuse_dim=128)
        seg_model = InceptionSegModel(backbone, head)
    
    seg_model.train()
    seg_model.backbone.eval()

    print("\n", "unfrozen model's layer name", [f"{n}" for n, p in seg_model.named_parameters() if p.requires_grad], "\n")

    return Modelsegmodel("inception_v3", dims, seg_model, n_classes)


def build_model(arch: str, linear_prob: bool, dims: int, n_classes: int, model_dir: Optional[str] = None,lock_backbone=True) ->  Modelsegmodel:
    """Integrated function to build various models.

    Args:
        arch: Architecture name ('cmpsd', 'DPT', 's_tinyvit', 's_tinyvittimm', 'inception_v3')
        linear_prob: Whether to use a linear probe as the segmentation head.
        dims: Input dimensions (2 or 3).
        n_classes: Number of output classes.
        model_dir: Optional directory for model weights/pretrained models.

    Returns:
        A dictionary with <arch> as the key and the Modelsegmodel object as the value.
    """
    if arch == "cmpsd":
        model = build_cnn_seg(dims, n_classes, model_dir=model_dir,linear_prob=linear_prob,lock_backbone=lock_backbone)
    elif arch == "cmpsd_old":
        model = build_old_cnn_seg(dims, n_classes, model_dir=model_dir,linear_prob=linear_prob,lock_backbone=lock_backbone)
    elif arch in ['dpt', 'DPT']:
        model = build_dpt(dims, n_classes, model_dir=model_dir, linear_prob=linear_prob)
    elif arch == "s_tinyvit":
        model = build_tinyvit_dpt(dims, n_classes, model_dir=model_dir, linear_prob=linear_prob,lock_backbone=lock_backbone)
    elif arch == "s_tinyvittimm":
        model = build_tinyvittimm_dpt(dims, n_classes, model_dir=model_dir, linear_prob=linear_prob)
    elif arch == "inception_v3":
        model = build_inception_v3(dims, n_classes, linear_prob=linear_prob)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model 


