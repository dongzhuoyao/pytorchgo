# Author: Tao Hu <taohu620@gmail.com>

import torch
from pytorchgo.utils import logger
from termcolor import colored
from tabulate import tabulate
import warnings, os, sys

if sys.version_info[0] >= 3:
    #logger.warn("use reduce from functools...")
    from functools import reduce


def set_gpu(gpu):
    if not isinstance(gpu, str):
        gpu = str(gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    logger.warn("pytorchgo set gpu: {}".format(gpu))


def model_summary(model_list):
    if not isinstance(model_list, list):
        model_list = [model_list]

    from operator import mul

    for model in model_list:
        data = []
        trainable_param_num = 0
        all_param_num = 0
        for key, value in model.named_parameters():
            data.append([key, list(value.size()),value.requires_grad, value.dtype, value.device, value.is_leaf, str(value.grad_fn)])
            _num = reduce(mul, list(value.size()), 1)
            all_param_num += _num
            if value.requires_grad:
                trainable_param_num += _num
        table = tabulate(data, headers=["name", "shape","requires_grad", "dtype", "device", "is_leaf", "grad_fn"])
        logger.warning(" Arg Parameters: #param={}, #param(trainable) = {}".format(all_param_num, trainable_param_num))
        logger.info(colored(
                "Model Summary",
                "cyan",
            ))
        logger.info(
            "\n\n"+
            table
        )

        logger.info(model)
        return all_param_num, trainable_param_num


def optimizer_summary(optim_list):
    if not isinstance(optim_list, list):
        optim_list = [optim_list]

    from operator import mul

    for optim in optim_list:

        assert isinstance(optim, torch.optim.Optimizer), ValueError(
            "must be an Optimizer instance"
        )
        data = []
        param_num = 0
        for group_id, param_group in enumerate(optim.param_groups):
            lr = param_group["lr"]
            weight_decay = param_group["weight_decay"]
            for id, param in enumerate(param_group["params"]):
                requires_grad = param.requires_grad
                shape = list(param.data.size())
                param_num += reduce(mul, shape, 1)
                data.append(
                    [group_id, id, shape, lr, weight_decay, requires_grad]
                )
        table = tabulate(
            data,
            headers=[
                "group",
                "id",
                "shape",
                "lr",
                "weight_decay",
                "requires_grad",
            ],
        )
        logger.info(
            colored(
                "Optimizer Summary, Optimzer Parameters: #param={} \n".format(
                    param_num
                ),
                "cyan",
            )
            + table
        )


def step_scheduler(optimizer, current_epoch, lr_schedule, net_name):
    """
    Function to perform step learning rate decay
    Args:
        optimizer: Optimizer for which step decay has to be applied
        epoch: Current epoch number
    """
    warnings.warn(
        "please use step_scheduler in pytorchgo.utils.learning_rate",
        DeprecationWarning,
    )
    previous_lr = optimizer.param_groups[0]["lr"]
    for (e, v) in lr_schedule:
        if current_epoch == e - 1:  # epoch start from 0
            logger.warn(
                "epoch {}: {} lr changed from: {} to {}".format(
                    current_epoch, net_name, previous_lr, v
                )
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = v

    return optimizer
