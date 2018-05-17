# Author: Tao Hu <taohu620@gmail.com>
from . import logger

def model_summary(model):
    logger.info(model)


def step_scheduler(optimizer, epoch):
    """
    Function to perform step learning rate decay
    Args:
        optimizer: Optimizer for which step decay has to be applied
        epoch: Current epoch number
    """

    decay_factor = 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_factor

    return optimizer