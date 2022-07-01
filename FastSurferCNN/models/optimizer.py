import torch


def get_optimizer(model, cfg):
    if cfg.OPTIMIZER.OPTIMIZING_METHOD == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            momentum=cfg.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
            dampening=cfg.OPTIMIZER.DAMPENING,
            nesterov=cfg.OPTIMIZER.NESTEROV,
        )
    elif cfg.OPTIMIZER.OPTIMIZING_METHOD == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )

    elif cfg.OPTIMIZER.OPTIMIZING_METHOD == 'adamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )

    elif cfg.OPTIMIZER.OPTIMIZING_METHOD == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,  # learning rate
            momentum=0.6,  # 0.95,  # momentum factor
            alpha=0.90,  # smoothing constant (Discounting factor for the history/coming gradient)
            eps=1e-10,  # term added to the denominator to improve numerical stability
            weight_decay=1e-4,  # 0,  # weight decay (L2 penalty)
            centered=False  # if True, compute the centered RMSProp (gradient normalized by estimation of its variance)
        )
    else:
        raise NotImplementedError(f"{cfg.OPTIMIZER.OPTIMIZING_METHOD}"
                                  f" optimizer is not supported")