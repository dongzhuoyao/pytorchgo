# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import SimAnnotationTransform, SimDetection, Sim_CLASSES, detection_collate

from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from IPython import embed
from log import log
import numpy as np
import time
from tqdm import tqdm
from pytorchgo.utils import logger

is_debug = 0
iterations = 40000
stepvalues = (30000, 40000)#(60000, 80000, 100000)
validate_per = 5000

num_classes = 2

if is_debug == 1:
    validate_per = 5
    iterations = 100
    stepvalues = (30, 60, 100)
    validate_per = 10

train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
means = (104, 117, 123)  # only support voc now


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

logger.auto_set_dir()

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--dim', default=300, type=int, help='Size of the input image, only support 300 or 512')
parser.add_argument('-d', '--dataset', default='SIM',help='VOC or COCO dataset')

parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')# choices=[16, 32]
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=iterations, type=int, help='Number of training iterations')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--gpu', default=1, type=int, help='gpu')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

from pytorchgo.utils.pytorch_utils import set_gpu
set_gpu(args.gpu)


logger.info(args)



start_iter = 0



ssd_net = build_ssd('train', args.dim, num_classes)
net = ssd_net

"""
if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True
"""

if args.resume:
    logger.info('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
    start_iter = int(args.resume.split('/')[-1].split('.')[0].split('_')[-1])
else:
    vgg_weights = torch.load(args.basenet)
    logger.info('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)
    start_iter = 0

if args.cuda:
    net = net.cuda()


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    logger.info('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, args.dim, 0.5, True, 0, True, 3, 0.5, False, args.cuda)


def train():
    net.train()
    logger.info('Loading Dataset...')

    dataset=SimDetection(transform=SSDAugmentation(
        args.dim, means))


    epoch_size = len(dataset) // args.batch_size
    logger.info('Training SSD on {}'.format(dataset.name))
    logger.info("epoch size: {}".format(epoch_size))
    step_index = 0
    batch_iterator = None
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)

    lr=args.lr
    epoch = 0
    for iteration in tqdm(range(start_iter, args.iterations), desc="epoch {}/{} training".format(epoch, args.iterations//epoch_size)):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
            epoch += 1
        if iteration in stepvalues:
            step_index += 1
            old_lr = lr
            lr = adjust_learning_rate(optimizer, args.gamma, step_index)
            logger.info("iter {}, change lr from {:.8f} to {:.8f}".format(iteration, old_lr, lr))

        images, targets = next(batch_iterator)
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        if iteration % 10 == 0:
            logger.info('''
                Timer: {:.5f} sec.\t LR: {:.7f}.\t Iter: {}.\t Loss_l: {:.5f}.\t Loss_c: {:.5f}. Loss: {:.5f}
                '''.format((t1-t0),lr,iteration,loss_l.data[0],loss_c.data[0],loss.data[0]))


        if iteration % validate_per == 0 and iteration > 0:
            logger.info('Saving state, iter={}'.format(iteration))
            torch.save(ssd_net.state_dict(), os.path.join(logger.get_logger_dir(),
                       'ssd-{}.pth'.format(repr(iteration))))
    torch.save(ssd_net.state_dict(),  os.path.join(logger.get_logger_dir(), 'ssd_{}.pth'.format(iteration)))
    logger.info("Congratulations..")

    
def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
