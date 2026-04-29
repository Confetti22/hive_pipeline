#%%
import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from config.load_config import load_cfg
from torchsummary import summary
import time
import math
import os

device ='cuda'
cfg_path = 'config/rm009_3d_one_stage.yaml'
args = load_cfg(cfg_path)
E5 = args.e5

args.data_path_dir = args.e5_data_path_dir if E5 else args.data_path_dir
args.valid_data_path_dir = args.e5_valid_data_path_dir if E5 else args.valid_data_path_dir
args.valid_msk_dir = args.e5_valid_msk_dir if E5 else args.valid_msk_dir


from lib.arch.ae import build_contrastive_model
from lib.core.optimizer import get_parameter_groups
model = build_contrastive_model(args).to(device) 
model.train()


#out_shape : B*C*D*H*W ---> need to squeeze
print(model)
summary(model,(1,*args.input_size))

from lib.datasets.contrastive_dataset import Contrastive_dataset_3d_one_stage
from lib.datasets.simple_segdataset import get_dataset, get_valid_dataset

train_dataset = Contrastive_dataset_3d_one_stage(args.data_path_dir,d_near=args.d_near,num_pairs=args.num_pairs,n_view=args.n_views,channel=args.img_channel,img_level = args.img_level) 

train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_per_gpu,shuffle= True,drop_last= True, pin_memory=True,num_workers=args.num_workers)


#roi: B*C*D*H*W  = B*1*D*H*W for rm009 boundary seg dataset
#mask: B*D*H*W   = B*1*H*W for rm009 boundary seg dataset  
valid_dataset = get_valid_dataset(
    data_path_dir= args.valid_data_path_dir,
    mask_path_dir=args.valid_msk_dir,
    use_ratio=0.08,
    normalize=True,
    make_3ch=False,
    shift_labels_to_zero=False
)
valid_loader = DataLoader(dataset=valid_dataset,batch_size=3,shuffle=False,drop_last=False,pin_memory=True,) 


exp_save_dir = 'runs/contrastive'
exp_name =f'onestage_batch{args.batch_per_gpu}_nview{args.n_views}_infoloss{args.infonce}_rm009_4um_mlp{args.mlp_filters}'


writer = SummaryWriter(log_dir=f'{exp_save_dir}/{exp_name}')

import os
import shutil
script_path = os.path.abspath(__file__)
shutil.copy2(script_path,f"{exp_save_dir}/{exp_name}/script.py")
shutil.copy2(cfg_path,f"{exp_save_dir}/{exp_name}/config.yaml")
print(f"config has been saved")



#load entire array into memory to accelate indexing feats
current = time.time()
# image only extend to 300 at y axis, and only take the half brain at right

optim_groups = get_parameter_groups(model, weight_decay=args.wd)
print(f"Separate parameters into weight decay and no weight decay groups")

optimizer = torch.optim.AdamW(
    optim_groups,
    lr=args.lr,
)
#%%

from lib.utils.augmentations import GPUAugmentations
from helper.contrastive_train_helper import valid_from_roi_dataloader

#only perform illumination and blur augmentation
augmentor = GPUAugmentations(size = None,affine=False,v=False).to(device)

model.eval()
model.off_proj()
valid_from_roi_dataloader(model,0,valid_loader,writer)
model.reset_proj()
model.train()

from lib.loss.cos_loss import get_loss
loss_fn = get_loss(args)

for epoch in range(args.start_epoch,args.num_epochs): 
    for it, batch in enumerate(train_loader):
        it = epoch * len(train_loader) +it
        batch = torch.cat(batch,dim=0)
        batch = batch.to(device)
        batch = augmentor(batch)

        optimizer.zero_grad()
        out = model(batch) 
        out = out.squeeze()
        loss,pos_cos,neg_cos = loss_fn(out)

        loss.backward() 
        optimizer.step() 

        # lr_scheduler1.step(loss)

        writer.add_scalar('Loss/train', loss.item(), it)
        writer.add_scalar('lr',optimizer.param_groups[0]["lr"], it)

        if math.isclose(pos_cos, 0.0, abs_tol=1e-9):
            writer.add_scalar('pos_cos',pos_cos, it)
            writer.add_scalar('neg_cos',neg_cos, it)

        print(f"epoch:{epoch}|batch{it}, loss: {loss:.4f}, pos_cos:{pos_cos:.4f}, neg_cos:{neg_cos:.4f}, lr:{optimizer.param_groups[0]["lr"]:.7f}")

    if (epoch) % args.valid_very_epoch ==0: 
        model.eval()
        model.off_proj()
        valid_from_roi_dataloader(model,epoch,valid_loader,writer)
        model.reset_proj()
        model.train()

    if (epoch+1) % args.valid_very_epoch*4 == 0:
        save_dir = f'{exp_save_dir}/{exp_name}'
        model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at epoch {epoch} to {model_path}")

# Optionally, save the final model
final_model_path = os.path.join(save_dir, 'model_final.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

writer.close()
                        










