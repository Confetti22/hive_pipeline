from dataclasses import dataclass
from torch import nn
import torch

@dataclass
class Modelsegmodel:
    name: str
    dims: int
    seg_model: nn.Module
    n_classes: int


def build_cmpsd(dims: int, n_classes: int,) -> Modelsegmodel:
    """Build cmpsd backbone + ConvSegHead.

    Args:
        dims: 2 or 3
        n_classes: number of output classes (incl. background)
        feat_channels: channels produced by backbone
    Returns:
        Modelsegmodel
    """
    level_key = 'l2'
    filters_map={'l1':[32,24,12,12],'l2':[64,32,24,12],'l3':[96,64,32,12]}
    cnn_filters_map ={'l1':[32],'l2':[32,64],'l3':[32,64,96]}
    cnn_kernler_size_map ={'l1':[5],'l2':[5,5],'l3':[5,5,3]}

    from config.load_config import load_cfg
    cfg = load_cfg('/home/confetti/e5_workspace/hive1/outs/contrastive_run_rm009/ae_mlp_rm009_v1/FEATl2_avg8_LOSSpostopk_numparis16384_batch4096_nview4_d_near6_shuffle20_cosdecay_valide_with_avgpool/config.yaml')
    cfg.in_channel = 1
    cfg.filters = cnn_filters_map[level_key] 
    cfg.kernel_size =cnn_kernler_size_map[level_key]
    cfg.mlp_filters = filters_map[level_key]
    cfg.last_encoder =False 
    cfg.avg_pool_size = [8,8,8]

    #todo: try different mlp weights at different epoch: 50, 200, 2000
    from lib.arch.ae_old import build_final_model,load_compose_encoder_dict
    cmpsd_model = build_final_model(cfg)
    cmpsd_model.eval()

    for param in cmpsd_model.parameters():
        param.requires_grad = False

    cnn_ckpt_pth = "/home/confetti/data/weights/ae_feats_nissel_v1_roi1_decaylr_e1600.pth"
    mlp_ckpt_pth = "/home/confetti/e5_workspace/hive1/outs/contrastive_run_rm009/ae_mlp_rm009_v1/FEATl2_avg8_LOSSpostopk_numparis16384_batch4096_nview4_d_near6_shuffle20_cosdecay_valide_with_avgpool/checkpoints/epoch_4000.pth"
    mlp_ckpt = torch.load(mlp_ckpt_pth)['model']
    load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_weight_dict=mlp_ckpt,dims=dims) #this pretrained model is 3d

    from lib.arch.ae import ConvMLP
    from lib.arch.segmodel import SimpleSegmodel
    seg_head = ConvMLP(filters=[12,12,n_classes],l2_norm=False,last_act=False,dims=dims).train() 
    seg_model = SimpleSegmodel(cmpsd_model,seg_head)

    cnn_ckpt = torch.load(cnn_ckpt_pth)
    print(f"\n\n{cnn_ckpt.keys()}= ")
    print(f"\n\n{mlp_ckpt.keys()}= ")
    print(f"\n\n{seg_model}")
    # summary(seg_model,(1,64,64,64))
     

    print("\n","unfrozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if  p.requires_grad],"\n")

    return Modelsegmodel("cmpsd", 3 ,seg_model,n_classes)


##### DPT ######
from transformers import AutoModel
from lib.arch.segdino import Dinov3HFBackbone,DPT

def build_dpt(dims: int, n_classes: int, ) -> Modelsegmodel:
    """Build DPT with DINOv3-like backbone + DPTHead.

    Notes:
        - For dims=3, we evaluate per-slice; backbone remains 2D.
        - seghead expects 2D features; for 3D ROI we loop slices externally.
    """

    model_dir = "/home/confetti/e5_workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m"# ViT-S/16 (patch=16)
    hf_backbone = AutoModel.from_pretrained(
        model_dir, local_files_only=True, output_hidden_states=True
    ).eval()
    backbone = Dinov3HFBackbone(hf_backbone)
    seg_model = DPT(nclass=n_classes,backbone=backbone)
    seg_model.train()

    #freeze backbone
    seg_model.lock_backbone()

    print("\n","unfrozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if  p.requires_grad],"\n")

    return Modelsegmodel("DPT", dims,seg_model,n_classes)



from lib.distill.student import TinyVitBackbone, TinyViTWithTaps, TinyViTWithTapsTimm
def build_tinyvit_dpt(dims:int, n_classes:int,) -> Modelsegmodel:


    model_dir = '/home/confetti/e5_workspace/hive1_pipeline/runs/_distill_cnn_test9_tinyvit_mullayer_nomixup_cosine_proj_t1779/student_epoch_010.pth'
    _model = TinyViTWithTaps( embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0).eval()
    # _model.register_taps([2,5,8,11])

    state_dict = torch.load(model_dir)
    result=_model.load_state_dict(state_dict)
    print(f"load tinyvit model state dict result: {result}")

    backbone  = TinyVitBackbone(model=_model) 

    seg_model = DPT(nclass=n_classes,backbone=backbone)
    seg_model.train()

    #freeze backbone
    seg_model.lock_backbone()

    print("\n","unfrozen model's layer name",[f"{n}" for n, p in seg_model.named_parameters() if  p.requires_grad],"\n")
    return Modelsegmodel("s_tinyvit", dims,seg_model,n_classes)





def build_tinyvittimm_dpt(dims:int, n_classes:int,) -> Modelsegmodel:

    model_dir = 'runs/_distill_cnn_test10_tinyvittimm_mullayer_nomixup_cosine_proj_t1779/student_epoch_070.pth'
    _model = TinyViTWithTapsTimm().eval()
    # _model.register_taps([2,5,8,11])

    state_dict = torch.load(model_dir)
    result=_model.load_state_dict(state_dict)
    print(f"load tinyvittimm model state dict result: {result}")

    backbone  = TinyVitBackbone(model=_model) 

    seg_model = DPT(nclass=n_classes,backbone=backbone)
    seg_model.train()

    #freeze backbone
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


######inception_v3 #######
from lib.arch.inception import InceptionBackbone,Inception_V3_Weights,InceptionSegHead,InceptionSegModel

def build_inception_v3(dims: int, n_classes: int) -> Modelsegmodel:
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

    head = InceptionSegHead(backbone.out_channels, n_classes=n_classes, proj_dim=128, fuse_dim=128)
    seg_model = InceptionSegModel(backbone, head)
    seg_model.train()
    seg_model.backbone.eval()

    print("\n", "unfrozen model's layer name", [f"{n}" for n, p in seg_model.named_parameters() if p.requires_grad], "\n")

    return Modelsegmodel("inception_v3", dims, seg_model, n_classes)


