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