import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise separable conv block with dilation support."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class FactorizedLargeKernelBlock(nn.Module):
    """
    Two 1D depthwise convolutions approximate a large k×k kernel with far fewer params.
    Striding is split across height/width to preserve an overall stride of `stride`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        stride: int = 2,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel for symmetric padding"
        pad = (kernel_size // 2) * dilation
        self.dw_width = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, pad),
            dilation=(1, dilation),
            groups=in_channels,
            bias=False,
        )
        self.dw_height = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(pad, 0),
            dilation=(dilation, 1),
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_width(x)
        x = self.dw_height(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class SmallReceptiveFieldCNN(nn.Module):
    """
    Compact CNN (~0.2M params) with a >128 px receptive field on the bottleneck features.
    Four stride-2 stages yield a 16× spatial reduction so the bottleneck aligns with ViT 14×14 tokens.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        super().__init__()
        self.stem = FactorizedLargeKernelBlock(
            in_channels, 32, kernel_size=15, stride=2, dilation=1
        )
        self.stage2 = DepthwiseSeparableBlock(
            32, 64, kernel_size=5, stride=2, dilation=2
        )
        self.stage3 = DepthwiseSeparableBlock(
            64, 96, kernel_size=5, stride=2, dilation=2
        )
        self.stage4 = DepthwiseSeparableBlock(
            96, 128, kernel_size=3, stride=2, dilation=4
        )
        # Extra dilated block bumps the receptive field beyond 128 px without extra downsampling.
        self.bottleneck = DepthwiseSeparableBlock(
            128, 128, kernel_size=3, stride=1, dilation=4
        )

        self.recon_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = x.shape[-2:]
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        bottleneck = self.bottleneck(x)
        recon = self.recon_head(bottleneck)
        recon = F.interpolate(recon, size=spatial, mode="bilinear", align_corners=False)
        return bottleneck, recon


def build_s_cnn() -> nn.Module:
    """
    Build the student CNN used for distillation.
    The architecture keeps parameters well under 2MB while giving the bottleneck a wide receptive field.
    """
    model = SmallReceptiveFieldCNN(in_channels=3, out_channels=3)
    # Sanity check: 2 MB / 4 bytes per float = 524288 params
    max_params = 2 * 1024 * 1024 // 4
    total_params = sum(p.numel() for p in model.parameters())
    assert (
        total_params <= max_params
    ), f"Student CNN too large ({total_params} params). Expected <={max_params}."

    return model
