import wandb
from pytorchgo.utils import logger
import numpy as np
from collections import deque

def wandb_logging(d, step, gpu, use_wandb=True, prefix="training:"):
    if gpu==0:
        if use_wandb:
            wandb.log(d, step=step)
        _str = "{} step={}\t".format(prefix, step)
        for k,v in d.items():
            _str += "{k}={v:.4f}\t".format(k=k,v=v)
        logger.info("GPU: {}, {}".format(gpu,_str))
        #print(_str)