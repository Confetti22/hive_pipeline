import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment", 
        data="my_data_dir",
        model="torchvision/resnet18",
        method="distillationv1",
        method_args={
            "teacher": "dinov3/vitb16",
            # Replace with your own url
            "teacher_url": "https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth<SOME-KEY>",
        }
    )