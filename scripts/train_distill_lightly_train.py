
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="runs/_ditill_lightly_train_test",
        data="/home/confetti/data/rm009/boundary_seg/new_boundary_seg_data/z_slices",
        model="torchvision/shufflenet_v2_x0_5",
        method="distillationv1",
        method_args={
            "teacher": "dinov3/vitb16",
            # 'teacher_weights':'/home/confetti/e5_workspace/hive1/models/facebook/dinov3-vits16-pretrain-lvd1689m/model.safetensors',
        },
        overwrite = True,
        epochs=500,
        
    )
