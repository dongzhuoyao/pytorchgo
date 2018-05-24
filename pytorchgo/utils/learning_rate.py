# Author: Tao Hu <taohu620@gmail.com>
from pytorchgo.utils import logger

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, base_lr, i_iter, total_iter, power=0.9):
    lr = lr_poly(base_lr, i_iter, total_iter, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

    return lr

def step_scheduler(optimizer, current_epoch, lr_schedule, net_name):
    """
    Function to perform step learning rate decay
    Args:
        optimizer: Optimizer for which step decay has to be applied
        epoch: Current epoch number
    """
    previous_lr = optimizer.param_groups[0]['lr']
    for (e, v) in lr_schedule:
        if current_epoch == e-1:#epoch start from 0
            logger.warn("epoch {}: {} lr changed from: {} to {}".format(current_epoch, net_name, previous_lr, v))
            for param_group in optimizer.param_groups:
                param_group['lr'] = v

    return optimizer