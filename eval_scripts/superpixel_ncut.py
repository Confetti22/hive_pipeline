#%%
import tifffile as tif
from scipy.ndimage import zoom
import torch
from pathlib import Path
import numpy as np
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
from lib.arch.ae_old import build_contrastive_model,load_compose_encoder_dict,build_encoder_model,load_encoder2encoder
from config.load_config import load_cfg
from helper.ncut_helper import segment_and_plot_from_feats 

from lib.utils.preprocess_img import preprocess_uint16_for_imagenet, to_cdhw
from torchsummary import summary
device ='cuda'
print(f'{os.getcwd()}=')
args = load_cfg('./config/t11_3d.yaml')
args.avg_pool_size = (8,8,8) 

cmpsd_model = build_contrastive_model(args)
cmpsd_model.eval().to(device)
cnn_ckpt_pth = '/home/confetti/data/weights/t11_3d_ae_best2.pth'
mlp_ckpt_pth ='/home/confetti/data/weights/t11_3d_mlp_best_new_format.pth'
load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_ckpt_pth,dims=args.dims)

encoder_model = build_encoder_model(args,dims=3) 
encoder_model.eval().to(device)
load_encoder2encoder(encoder_model,cnn_ckpt_pth)
summary(cmpsd_model, (1,64,1024,1024))
#%%

from config.load_config import load_cfg
cfg = load_cfg('/home/confetti/e5_workspace/hive1_pipeline/runs/contrastive/onestage_batch2028_nview2_infolossFalse_t1779_2um/config.yaml')
avg_pool_size = 4
cfg.dims = 3
cfg.avg_pool_size = [avg_pool_size]* 3

from lib.arch.ae import build_contrastive_model
contrast_model = build_contrastive_model(cfg)
contrast_model.eval()
contrast_model.off_proj() #discard projecitonhead in downstream task
model_dir = '/home/confetti/e5_workspace/hive1_pipeline/runs/contrastive/onestage_batch2028_nview2_infolossFalse_t1779_2um/model_epoch_100.pth'
cmpsd_ckpt = torch.load(model_dir)
contrast_model.load_state_dict(cmpsd_ckpt)
contrast_model = contrast_model.to(device)
print(f"load weight at {model_dir}")
for param in contrast_model.parameters():
    param.requires_grad = False
summary(contrast_model,(1,32,768,768))


#%%
import torchvision.models as models
from torchvision.models import Inception_V3_Weights

weights = Inception_V3_Weights.DEFAULT
incep_model = models.inception_v3(weights = weights, progress =True)
incep_model.eval()
incep_model.to(device)
print(incep_model)
summary(incep_model,(3,1536,1536))

#%%
import torch
from lib.utils.preprocess_img import pad_to_multiple_of_unit
import numpy as np
from helper.image_reader import Ims_Image
from scipy.ndimage import gaussian_filter
import tifffile as tif

ims_vol =Ims_Image('/home/confetti/e5_data/t1779/t1779.ims',channel=2)

roi_list = [['/home/confetti/data/t1779/scenes/hp_1536_1536_64.tif'],['/home/confetti/data/t1779/scenes/vii_1536_1536_64.tif'],['/home/confetti/data/t1779/scenes/visa2_1536_1536_64.tif']]
save_dir = '/home/confetti/e5_workspace/hive1_pipeline/results/figs'

for idx,path  in enumerate(roi_list):
    vol = tif.imread(path)
    vol = vol.astype(np.uint64)
    vol_2 = zoom(vol,(0.5,0.5,0.5),order=2)

    input2d = vol[32]
    normalized_img = (((input2d -input2d.min())/ (input2d.max() - input2d.min())) * 255).astype(np.uint8)
    rgb_img = np.stack([normalized_img] * 3, axis=-1)

    zoom_factor= 8
    print(f"{vol.shape= }")
    vol = pad_to_multiple_of_unit(vol,unit=zoom_factor) 
    print(f"{vol.shape= }")


    # blurred_image = gaussian_filter(vol[32,], sigma=4,mode='reflect')
    # image = vol[32,:,:] 

    input3d = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(device) #add batch and channel dim B*C*H*W
    input3d_2 = torch.from_numpy(vol_2).unsqueeze(0).unsqueeze(0).float() #add batch and channel dim B*C*H*W
    processed =  preprocess_uint16_for_imagenet(
                input3d_2,
                make_3ch=False,
            )
    processed_input = processed.squeeze(0).squeeze(0).to(device)
    print(f"{processed_input.shape=}")

    mlp_out = cmpsd_model(input3d).cpu().detach().squeeze().numpy()
    cnn_out = encoder_model(input3d).cpu().detach().squeeze().numpy()
    contrast_one_out = contrast_model(processed_input).cpu().detach().squeeze().numpy()

    print(f"{mlp_out.shape= }")
    print(f"{cnn_out.shape= }")
    print(f"{contrast_one_out.shape= }")
    # C,H,W = mlp_out.shape

    mlp_feats_map = mlp_out[:,:,:]
    mlp_feats_map = np.moveaxis(mlp_feats_map, 0,-1)
    cnn_feats_map = cnn_out[:,:,:]
    cnn_feats_map = np.moveaxis(cnn_feats_map, 0,-1)
    contrast_feats_map = np.moveaxis(contrast_one_out,0,-1)


    from confettii.feat_extract import get_feature_list,get_feature_map,TraverseDataset2d,TraverseDataset3d,TraverseDataset3d_overlap
    from torch.utils.data import DataLoader
    inception_extract_layer_name = 'avgpool'
    two_dim_dataset = TraverseDataset2d(rgb_img,stride=16,win_size=128)
    out_shape = two_dim_dataset._get_sample_shape()
    two_dim_loader = DataLoader(two_dim_dataset,batch_size=512,shuffle=None,drop_last=False) 

    feats_list = get_feature_list('cuda',incep_model,two_dim_loader,extract_layer_name=inception_extract_layer_name)
    from sklearn.decomposition import PCA
    from skimage import io,  color,  exposure

    import numpy as np
    compactness=1 
    sigma = 0.0001

    # PCA for visualization
    pca = PCA(n_components=12)
    inception_feats_map = pca.fit_transform(feats_list).reshape(*out_shape,12)
    print(f"end prepare for inception")

    # segment_and_plot_from_feats(inception_feats_map,rgb_img,save_dir=save_dir,tag=f'inception_{idx}')

    segment_and_plot_from_feats(mlp_feats_map,rgb_img,n_segments=100,save_dir=save_dir,tag=f"contrastive_two_stage_{idx}_2")
    # segment_and_plot_from_feats(contrast_feats_map,rgb_img,n_segments=100,save_dir=save_dir,tag=f"contrastive_one_stage_e150_{idx}")
    # segment_and_plot_from_feats(cnn_feats_map,rgb_img)
#%%
# # Normalize to [0,1] for processing
# image = exposure.rescale_intensity(image, in_range='image', out_range=(0, 1))
# print(f"Image dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")

# # Step 2: Gaussian blur (still in float)
# blurred = gaussian_filter(image, sigma=12)
# blurred = image

# # Step 3: Superpixel segmentation (SLIC expects RGB-like input)
# image_rgb = np.stack([blurred]*3, axis=-1)
# segment_and_plot_from_feats(image_rgb,rgb_img)


#%%
# %%
