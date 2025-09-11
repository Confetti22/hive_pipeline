import torch.nn as nn
import torch
import os



def get_norm_nd(norm: str, out_channels: int, dim: int = 3, bn_momentum: float = 0.1) -> nn.Module:
    """Return a normalization layer for N-D convolution.

    Args:
        norm (str): One of ['bn', 'sync_bn', 'in', 'gn', 'none'].
        out_channels (int): Number of output channels.
        dim (int): dims of data (1, 2, or 3).
        bn_momentum (float): Momentum for BatchNorm or SyncBatchNorm.

    Returns:
        nn.Module: Normalization layer.
    """
    norm = norm.lower()
    assert norm in ["bn", "sync_bn", "in", "gn", "none"], f"Unknown normalization type: {norm}"
    assert dim in [1, 2, 3], f"Unsupported dims: {dim}"

    norm_layers = {
        "bn": [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d],
        "sync_bn": [nn.SyncBatchNorm] * 3,
        "in": [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d],
    }

    if norm in norm_layers:
        NormLayer = norm_layers[norm][dim - 1]
        return NormLayer(out_channels, momentum=bn_momentum)

    if norm == "gn":
        assert out_channels % 8 == 0, "GroupNorm requires out_channels divisible by 8"
        return nn.GroupNorm(8, out_channels)

    return nn.Identity()




def get_activation(act: str = 'relu') -> nn.Module:
    """Return an activation layer.

    Args:
        act (str): One of ['relu', 'leaky_relu', 'elu', 'gelu', 'swish', 'efficient_swish', 'none'].

    Returns:
        nn.Module: Activation layer.
    """
    act = act.lower()
    assert act in ["relu", "leaky_relu", "elu", "gelu", "swish", "efficient_swish", "none"], \
        f"Unknown activation type: {act}"

    activation_dict = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
        "elu": nn.ELU(inplace=True),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),  # swish = SiLU
        "efficient_swish": nn.SiLU(),
        "none": nn.Identity()
    }

    return activation_dict[act]


def conv_nd_norm_act(in_channels, out_channels,
                     kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True,
                     dim=3, trans=False,
                     pad_mode='replicate', norm_mode='bn', act_mode='relu',
                     return_list=False,
                     output_padding = 0):

    assert dim in [1, 2, 3], "Only 1D, 2D, or 3D convolutions are supported"

    # Padding mode compatibility
    if pad_mode not in ['zeros', 'reflect', 'replicate', 'circular']:
        pad_mode = 'zeros'


    # Dynamically pick Conv or ConvTranspose
    if trans:
        Conv = [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d][dim - 1]
        conv_layer = Conv(
        in_channels, out_channels,
        kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation,
        groups=groups, bias=bias,
        padding_mode='zeros',  # transposed conv only supports 'zeros'
        output_padding= output_padding,
    )
    else:
        Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1]
        conv_layer = Conv(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias,
            padding_mode=pad_mode, 
        )

    norm_layer = get_norm_nd(norm_mode, out_channels, dim)
    act_layer = get_activation(act_mode)

    layers = [conv_layer, norm_layer, act_layer]

    return layers if return_list else nn.Sequential(*layers)


def make_block(in_ch, out_ch, ks, stride, block_type, dim, trans, shared_kwargs,padding=0,output_padding=0 ):
    def conv(in_c, out_c):
        return conv_nd_norm_act(
            in_c, out_c, ks,
            stride if in_c == in_ch else 1,
            padding,
            dim=dim,
            trans=trans if in_c == in_ch else False,
            output_padding=output_padding,
            **shared_kwargs
        )

    if block_type == 'double':
        return nn.Sequential(conv(in_ch, out_ch), conv(out_ch, out_ch))
    elif block_type == 'triple':
        return nn.Sequential(conv(in_ch, out_ch), conv(out_ch, out_ch), conv(out_ch, out_ch))
    else:  # 'single'
        return nn.Sequential(conv(in_ch, out_ch))

# --------------------------------------------
#  Base AutoEncoder Class (ND)
# --------------------------------------------

class EncoderND(nn.Module):
    """
    update 2025/08/14
    Encoder with N down_layers. Downsampling per stage is controlled by `downsample_strategy`:
      - 'conv_stride': first conv in the block has stride=2 (no padding).
      - 'max_pool':    all convs use stride=1 (no padding), then MaxPool(k=2,s=2).

    Everything (including the former conv_in and optional last 1×1 conv) is a down_layer.
    """
    def __init__(self, in_channel, filters, kernel_size, dims=3,
                 pad_mode='reflect', act_mode='elu', norm_mode='gn',
                 block_type='double',
                 downsample_strategy='conv_stride'):  # 'conv_stride' or 'max_pool'
        super().__init__()
        assert downsample_strategy in ('conv_stride', 'max_pool'), \
            "downsample_strategy must be 'conv_stride' or 'max_pool'"

        self.dim =dims 
        self.depth = len(filters)
        self.downsample_strategy = downsample_strategy

        Pool = nn.MaxPool3d if dims== 3 else nn.MaxPool2d
        Conv = nn.Conv3d if dims== 3 else nn.Conv2d

        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode
        }

        self.down_layers = nn.ModuleList()

        # ---- Stage 0: former conv_in, now a down_layer (single block, no padding) ----
        k0 = kernel_size[0]

        if self.downsample_strategy == 'conv_stride':
            stage0 = make_block(in_channel, filters[0], k0, stride=2,
                                block_type=block_type, dim=dims, trans=False,
                                shared_kwargs=self.shared_kwargs)
        else:
            stage0_block = make_block(in_channel, filters[0], k0, stride=1,
                                      block_type=block_type, dim=dims, trans=False,
                                      shared_kwargs=self.shared_kwargs)
            stage0 = nn.Sequential(stage0_block, Pool(kernel_size=2, stride=2))

        self.down_layers.append(stage0)

        # ---- Stages 1..depth-1 ----
        for i in range(self.depth - 1):
            ks = kernel_size[min(i + 1, len(kernel_size) - 1)]

            if self.downsample_strategy == 'conv_stride':
                block = make_block(filters[i], filters[i + 1], ks, stride=2,
                                   block_type=block_type, dim=dims, trans=False,
                                   shared_kwargs=self.shared_kwargs)
                stage = block
            else:
                block = make_block(filters[i], filters[i + 1], ks, stride=1,
                                   block_type=block_type, dim=dims, trans=False,
                                   shared_kwargs=self.shared_kwargs)
                if i == self.depth - 1 -1:
                    stage = block
                else:
                    stage = nn.Sequential(block, Pool(kernel_size=2, stride=2))

            self.down_layers.append(stage)


    def forward(self, x):
        for layer in self.down_layers:
            x = layer(x)
        return x


class DecoderND(nn.Module):
    """
    Decoder with upsampling via ConvTranspose (stride=2), no padding anywhere.
    All stages (including the final out stage) are appended to self.up_layers.
    
    update 2025/09/03 add parameter:last_layer_act='none',  controls whether to use activation at the last layer 
    (for raw input image reconstruciton task, a linear layer is better)

    """
    def __init__(self, out_channel, filters, kernel_size, dims=3,
                 pad_mode='reflect', act_mode='elu', norm_mode='gn',
                 block_type='double',output_padding =0,last_layer_act='none'):
        super().__init__()
        self.dim = dims
        self.depth = len(filters)

        ConvTrans = nn.ConvTranspose3d if dims == 3 else nn.ConvTranspose2d

        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode
        }

        self.up_layers = nn.ModuleList()

        # Stages: depth-1 .. 1 (each: TConv stride=2 (no padding) + optional convs stride=1)
        for i in reversed(range(self.depth - 1)):
            ks = kernel_size[min(i + 1, len(kernel_size) - 1)]
            block = make_block(
                filters[i + 1], filters[i],
                ks,                    # kernel size
                stride=2,        # upsample here
                block_type=block_type,
                dim=dims,
                trans=True,            # first conv is transposed conv
                output_padding=output_padding,
                shared_kwargs=self.shared_kwargs
            )
            self.up_layers.append(block)

        # Final out stage as an up_layer (ConvTranspose to out_channel, no padding)
        block = make_block(
                filters[0], out_channel,
                kernel_size[0],                    # kernel size
                stride=2,        # upsample here
                block_type=block_type,
                dim=dims,
                trans=True,            # first conv is transposed conv
                output_padding=1,
                shared_kwargs = {
                    'pad_mode': pad_mode,
                    'act_mode': last_layer_act,
                    'norm_mode': norm_mode}

            )
        self.up_layers.append(block)
    def forward(self, x):
        for layer in self.up_layers:
            x = layer(x)
        return x



class BaseAutoEncoderND(nn.Module):
    def __init__(self, in_channel, out_channel, filters, kernel_size, dims,
                 pad_mode='reflect', act_mode='elu', norm_mode='none', block_type='single',downsample_strategy='max_pool',last_layer_act = 'none' ,return_bottle_neck=True):
        super().__init__()
        self.encoder = EncoderND(in_channel, filters, kernel_size, dims,
                                 pad_mode, act_mode, norm_mode, block_type,downsample_strategy)
        self.decoder = DecoderND(out_channel, filters, kernel_size, dims,
                                 pad_mode, act_mode, norm_mode, block_type,last_layer_act=last_layer_act)
        self.return_bottle_neck = return_bottle_neck

    def forward(self, x):
        bottle_neck = self.encoder(x)
        cnn_out = self.decoder(bottle_neck)
        if self.return_bottle_neck:
            return bottle_neck,cnn_out
        else:
            return cnn_out

# --------------------------------------------
# MLP 
# --------------------------------------------


class MLP(nn.Module):
    def __init__(self, filters=[24, 18, 12, 8]):
        super(MLP, self).__init__()
        
        layers = []
        for in_features, out_features in zip(filters[:-1], filters[1:]):
            layers.append(nn.Linear(in_features, out_features))
        
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)  # Last layer, no activation
        return x / x.norm(p=2, dim=-1, keepdim=True)

## 1*1 conv version of the above MLP

class ConvMLP(nn.Module):
    def __init__(self, filters=[24, 18, 12, 8], dims=2,l2_norm=True,last_act=False):
        super(ConvMLP, self).__init__()

        assert dims in [2, 3], "dims must be 2 or 3"
        Conv = nn.Conv2d if dims == 2 else nn.Conv3d

        layers = []
        for in_channels, out_channels in zip(filters[:-1], filters[1:]):
            layers.append(Conv(in_channels, out_channels, kernel_size=1))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dims = dims  # Save dims for later use (e.g., normalization)
        self.l2_norm = l2_norm
        self.last_act = last_act

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)  # Last layer, no activation

        if self.last_act:
            x = self.relu(x)

        # L2 normalization across channel dim
        if self.l2_norm:
            x = x / x.norm(p=2, dim=1, keepdim=True)
        return x

class ComposedModel(nn.Module):
    def __init__(self, in_channel,cnn_filters, kernel_size,dims,mlp_filters, 
                 pad_mode='reflect', act_mode='elu', norm_mode='gn', block_type='double',downsample_strategy='conv_stride'):
        super().__init__()
        self.cnn_encoder = EncoderND(in_channel, cnn_filters, kernel_size, dims,
                                 pad_mode, act_mode, norm_mode, block_type,downsample_strategy=downsample_strategy)
        self.mlp_encoder = ConvMLP(mlp_filters,dims)

    def forward(self, x):
        x = self.cnn_encoder(x) # B*C*H*W --> B*H*W*C --> (B*H*W)*C
        # x = x.permute(0, 2, 3, 1)
        # x = x.reshape(-1, x.shape[-1])
        x = self.mlp_encoder(x)
        return x

MODEL_MAP = {
    'ae': BaseAutoEncoderND,
    'encoder': EncoderND,
}


def build_autoencoder_model(args):
    """Build an autoencoder with variant-specific kwargs.

    'ae' uses the classic BaseAutoEncoderND; 'ae_1' uses
    BaseAutoEncoderND_1 and may take `downsample_strategy`, `last_layer_act`,
    and `return_bottle_neck`.
    """

    model_arch = args.model_name
    assert model_arch in MODEL_MAP.keys(), f"Unknown model_name: {model_arch}"

    kwargs = {
        'in_channel': args.in_channel,
        'out_channel': args.out_channel,
        'filters': args.filters,
        'kernel_size': args.kernel_size,
        'pad_mode': args.pad_mode,
        'act_mode': args.act_mode,
        'norm_mode': args.norm_mode,
        'block_type': args.block_type,
        'downsample_strategy':args.downsample_strategy,
        'last_layer_act': args.last_layer_act,
        'return_bottle_neck': args.return_bottle_neck,
        'dims': args.dims,
    }

    model = MODEL_MAP[model_arch](**kwargs)
    print('model: ', model.__class__.__name__)

    return model


def build_encoder_model(args, dims):
    """Build only the encoder module using the same parameters as build_autoencoder_model.
    This function instantiates the autoencoder specified by `args.model_name` with
    the same kwargs as `build_autoencoder_model`, then returns only its `encoder`.
    Notes:
    - The `dims` parameter is kept for backward compatibility but is unused,
      since `args.dims` is already consumed by the autoencoder builder.
    - If you previously depended on constructing the standalone encoder via
      `args.encoder_model_name`, that path is now superseded by taking the
      encoder from the autoencoder to guarantee exact parity.
    """
    # Reuse the autoencoder factory to ensure exact parity of kwargs/architecture
    args.dims = dims  
    ae = build_autoencoder_model(args)
    encoder = ae.encoder
    print('model: ', encoder.__class__.__name__)
    return encoder



def build_cmpsd_model(args):
    kwargs = {
        'in_channel': args.in_channel,
        'cnn_filters': args.filters,
        'kernel_size': args.kernel_size,
        'pad_mode': args.pad_mode,
        'act_mode': args.act_mode,
        'norm_mode': args.norm_mode,
        'block_type': args.block_type,
        'dims':args.dims,
        'mlp_filters':args.mlp_filters,
        'downsample_strategy':args.downsample_strategy,

    }
    model = ComposedModel(**kwargs)
    return model


class semantic_seg(nn.Module):
    def __init__(self, in_channel,out_channel,filters, kernel_size,dims,mlp_filters, 
                pad_mode='reflect', act_mode='elu', norm_mode='gn', block_type='double',downsample_strategy='max_pool'):
        super().__init__()
        kwargs ={
            'in_channel': in_channel, 
            'out_channel': out_channel,
            'filters':filters, 
            'kernel_size': kernel_size, 
            'dims':dims,                 
            'pad_mode':pad_mode, 
            'act_mode':act_mode,
            'norm_mode':norm_mode, 
            'block_type':block_type,
            'downsample_strategy': downsample_strategy,
        }
        self.cnn_module = BaseAutoEncoderND(**kwargs)
        self.mlp_module = ConvMLP(mlp_filters,dims,l2_norm=False)

    def forward(self, x):
        bottle_neck,cnn_out = self.cnn_module(x) # B*C*H*W --> B*H*W*C --> (B*H*W)*C
        mlp_out = self.mlp_module(cnn_out)
        return bottle_neck,mlp_out

def build_semantic_seg_model(args):
    kwargs = {
        'in_channel': args.in_channel,
        'out_channel': args.out_channel,
        'filters': args.filters,
        'kernel_size': args.kernel_size,
        'pad_mode': args.pad_mode,
        'act_mode': args.act_mode,
        'norm_mode': args.norm_mode,
        'block_type': args.block_type,
        'dims':args.dims,
        'mlp_filters':args.mlp_filters,
        'downsample_strategy': args.downsample_strategy,
    }
    model = semantic_seg(**kwargs)
    return model
    


def modify_key(weight_dict,source,target):
    new_weight_dict = {}
    for key, value in weight_dict.items():
        new_key = key.replace(source,target)
        new_weight_dict[new_key] = value
    return new_weight_dict


def delete_key(weight_dict,pattern_lst:tuple):
    new_weight_dict = {k: v for k, v in weight_dict.items() if not k.startswith(pattern_lst)}
    return new_weight_dict 

def load_ae2encoder(model,ckpt_pth):
    ckpt = torch.load(ckpt_pth)
    removed_module_dict = modify_key(ckpt['model'],source='module.encoder.',target='')
    load_result = model.load_state_dict(removed_module_dict, strict=False)

    missing = load_result.missing_keys
    unexpected = load_result.unexpected_keys

    if not missing and not unexpected:
        print("✅ All weights loaded successfully.")
    else:
        print("⚠️ Some weights were not loaded exactly:")
        if missing:
            print(f"   • Missing keys ({len(missing)}):\n     {missing}")
        if unexpected:
            print(f"   • Unexpected keys ({len(unexpected)}):\n     {unexpected}")

    return load_result

def modify_key(weight_dict, source, target):
    return {k.replace(source, target): v for k, v in weight_dict.items()}
def delete_key(weight_dict, pattern_lst: tuple):
    return {k: v for k, v in weight_dict.items() if not k.startswith(pattern_lst)}


def _extract_state_dict(ckpt_or_path):
    """Return a raw state_dict from a checkpoint file or mapping.
    Supports:
    - Path to AE trainer checkpoint with keys {'epoch','model','optim'/'optimizer'}
    - Direct state_dict (possibly from DataParallel/DistributedDataParallel)
    """
    if isinstance(ckpt_or_path, (str, bytes, os.PathLike)):
        ckpt = torch.load(ckpt_or_path, map_location='cpu')
    else:
        ckpt = ckpt_or_path
    if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        return ckpt['model']
    return ckpt


def _state_dict_for_encoder(raw_sd: dict) -> dict:
    """Filter and normalize keys so they fit an encoder module.
    Handles keys like:
      - 'module.encoder.xxx' (DDP AE) → 'xxx'
      - 'encoder.xxx'        (AE)     → 'xxx'
      - 'module.xxx'         (DDP encoder-only) → 'xxx'
      - 'decoder.xxx' keys are dropped entirely
      - bare 'xxx' are kept (encoder-only saved w/o wrappers)
    """
    out = {}
    for k, v in raw_sd.items():
        if k.startswith('module.encoder.'):
            out[k[len('module.encoder.'):]] = v
        elif k.startswith('encoder.'):
            out[k[len('encoder.'):]] = v
        elif k.startswith('module.decoder.') or k.startswith('decoder.'):
            # skip decoder weights when loading into encoder
            continue
        elif k.startswith('module.'):
            # encoder-only saved under DataParallel
            out[k[len('module.'):]] = v
        else:
            # bare keys, assume encoder-only dict
            out[k] = v
    return out

def _report_load_result(tag: str, result: torch.nn.modules.module._IncompatibleKeys):
    missing = result.missing_keys
    unexpected = result.unexpected_keys
    if not missing and not unexpected:
        print(f"{tag}: ✅ All weights loaded successfully.")
    else:
        print(f"{tag}: ⚠️ Some weights were not loaded exactly:")
        if missing:
            print(f"   • Missing keys ({len(missing)}):\n     {missing}")
        if unexpected:
            print(f"   • Unexpected keys ({len(unexpected)}):\n     {unexpected}")

def load_encoder2encoder(model, ckpt):
    """Load an encoder module from an encoder-only or AE checkpoint.
    Accepts either a path or a raw mapping. Keys are normalized to fit the
    provided encoder module. If an AE checkpoint is passed, only the encoder
    subset is used.
    """
    raw_sd = _extract_state_dict(ckpt)
    norm_sd = _state_dict_for_encoder(raw_sd)
    result = model.load_state_dict(norm_sd, strict=False)
    _report_load_result('load_encoder2encoder', result)
    return result

def load_ae2encoder(model, ckpt):
    """Alias of load_encoder2encoder for clarity in call sites."""
    return load_encoder2encoder(model, ckpt)


def load_mlpencoder_dict(model,ckpt_pth):
    ckpt = torch.load(ckpt_pth)
    #remove any 'module.' keywords if exist in weights_pth and remove unwanted layers
    result = model.load_state_dict(ckpt,strict=False)
    _report_load_result('load_encoder2encoder', result)    

    return result


def load_mlp_ckpt_to_convmlp(convmlp_model, mlp_ckpt_pth=None, mlp_weight_dict=None, dims=2):
    if mlp_ckpt_pth is not None:
        mlp_ckpt = torch.load(mlp_ckpt_pth)
    elif mlp_weight_dict is not None:
        mlp_ckpt = mlp_weight_dict
    else:
        raise ValueError("Either 'mlp_ckpt_pth' or 'mlp_weight_dict' must be provided.")

    conv_state_dict = convmlp_model.state_dict()
    new_state_dict = {}

    linear_idx = 0
    for name, param in conv_state_dict.items():
        if 'weight' in name:
            linear_w = mlp_ckpt[f'layers.{linear_idx}.weight']  # shape: [out, in]
            if dims ==3:
                new_w = linear_w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape: [out, in, 1, 1]
            else:
                new_w = linear_w.unsqueeze(-1).unsqueeze(-1)  # shape: [out, in, 1, 1]
            new_state_dict[name] = new_w
        elif 'bias' in name:
            linear_b = mlp_ckpt[f'layers.{linear_idx}.bias']
            new_state_dict[name] = linear_b
            linear_idx += 1  # advance to next Linear layer
        else:
            raise ValueError(f'Unknown param name {name}')

    convmlp_model.load_state_dict(new_state_dict)
    print(f"load_mlp_ckpt_to_convmlp ✅all weiths loaded into convmlp successfully")

def load_compose_encoder_dict(cmodel,cnn_ckpt_pth,mlp_ckpt_pth=None,mlp_weight_dict=None,dims=2):
    cnn = cmodel.cnn_encoder
    mlp = cmodel.mlp_encoder
    load_encoder2encoder(cnn,cnn_ckpt_pth)
    # load_mlpencoder_dict(mlp,mlp_ckpt_pth)
    load_mlp_ckpt_to_convmlp(mlp,mlp_ckpt_pth,mlp_weight_dict,dims)
    




