import torch
import numpy as np
from dragon.utils.tools import logger

def get_optimizer(opt):
    if opt == 'SGD':
        return torch.optim.SGD
    elif opt == 'Adam':
        return torch.optim.Adam
    elif opt == 'Adamax':
        return torch.optim.Adamax
    elif opt == "AdamW":
        return torch.optim.AdamW
    elif opt == 'Adadelta':
        return torch.optim.Adadelta
    elif opt == 'Adagrad':
        return torch.optim.Adagrad
    elif opt == 'RMSprop':
        return torch.optim.RMSprop
    else:
        logger.error(f'Optimizer: {opt}, not found.')


def cyclical_lr(t, M, T, a0):
    """
    Compute cyclical learning rate as a function of the epoch number t: lr(t)
    t: Epoch number
    M: Number of explored local minima
    T: Number max of epochs
    a0: starting learning rate
    """
    cos_inner = np.pi * (t % (T // M))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    lr = float(a0 / 2 * cos_out)
    return lr


def configure_optimizers(model, opt_config):
    """
        Returns the optimizer to use.
        """
    opt = get_optimizer(opt_config['Optimizer'])
    kwargs = {
        "lr": opt_config['LearningRate']
    }
    if opt_config['Optimizer'] == "SGD":
        kwargs['momentum'] = 0.9
        kwargs['weight_decay'] = 1e-8
    # elif opt_config['Optimizer'] == "Adam":
    #    kwargs['weight_decay'] = 1e-8
    optimizer = opt(model.parameters(), **kwargs)
    if "Scheduler" in opt_config:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=opt_config['Scheduler'])
    else:
        scheduler = None
    return optimizer, scheduler
