# Author: Tao Hu <taohu620@gmail.com>
from . import logger
from termcolor import colored
from tabulate import tabulate


def model_summary(model):
    state_dict = model.state_dict().copy()
    params = filter(lambda p: p.requires_grad, model.parameters())

    data = []
    for key,value in state_dict.items():
        data.append([key,list(value.size())])
    table = tabulate(data, headers=['name', 'shape'])
    logger.info(colored("Arg Parameters: \n", 'cyan') + table)

    logger.info(model)


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