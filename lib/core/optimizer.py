import torch.optim as Opt

def get_optimizer(model, args):

    opt_fns = {
        'adam': Opt.Adam(model.parameters(), lr = args.lr_start, weight_decay=args.weight_decay),
        'sgd': Opt.SGD(model.parameters(), lr = args.lr_start, weight_decay=args.weight_decay),
        'adagrad': Opt.Adagrad(model.parameters(), lr = args.lr_start, weight_decay=args.weight_decay)
    }
    return opt_fns.get(args.optimizer, "Invalid Optimizer")

def get_parameter_groups(model, adapter_iterable=None, weight_decay=0.0005):
    # Parameters to exclude from weight decay
    no_decay = ["bias", "norm", "LayerNorm", "BatchNorm"]
    
    # Collect student parameters
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    # Collect adapter parameters
    if adapter_iterable:
        for adapter in adapter_iterable:
            if hasattr(adapter, "named_parameters"):
                for name, param in adapter.named_parameters():
                    if not param.requires_grad:
                        continue
                    if any(nd in name for nd in no_decay):
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)
    
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
