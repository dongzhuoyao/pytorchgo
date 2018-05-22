# Author: Tao Hu <taohu620@gmail.com>

import torch
from . import logger
from termcolor import colored
from tabulate import tabulate


def model_summary(model_list):
    if not isinstance(model_list, list):
        model_list = [model_list]

    for model in model_list:
        state_dict = model.state_dict().copy()
        params = filter(lambda p: p.requires_grad, model.parameters())

        data = []
        for key,value in state_dict.items():
            data.append([key,list(value.size())])
        table = tabulate(data, headers=['name', 'shape'])
        logger.info(colored("Arg Parameters: \n", 'cyan') + table)

        logger.info(model)

def optimizer_summary(optim_list):
    if not isinstance(optim_list, list):
        optim_list = [optim_list]

    from operator import mul
    for optim in optim_list:

        assert isinstance(optim, torch.optim.Optimizer),ValueError("must be an Optimizer instance")
        data = []
        param_num = 0
        for group_id, param_group in enumerate(optim.param_groups):
            lr = param_group['lr']
            weight_decay = param_group['weight_decay']
            for id, param in enumerate(param_group['params']):
                requires_grad = param.requires_grad
                is_volatile = param.volatile
                shape = list(param.data.size())
                param_num += reduce(mul, shape, 1)
                data.append([group_id, id, shape,lr,weight_decay,requires_grad,is_volatile])
        table = tabulate(data, headers=['group','id', 'shape', 'lr', 'weight_decay', 'requires_grad', 'volatile'])
        logger.info(colored("Optimzer Parameters: #param={} \n".format(param_num), 'cyan') + table)


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