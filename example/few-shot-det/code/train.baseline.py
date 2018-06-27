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
from data import AnnotationTransform, VOCDetection, VOCroot, VOC_CLASSES
from data import KittiLoader, AnnotationTransform_kitti,Class_to_ind

from data.FewShotDs import detection_collate

from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import numpy as np
import time
from pytorchgo.utils import logger

num_classes = 2

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--dim', default=512, type=int, help='Size of the input image, only support 300 or 512')
parser.add_argument('-d', '--dataset', default='VOC-FEWSHOT',help='VOC or COCO dataset')

parser.add_argument('--basenet', default='/home/hutao/lab/pytorchgo/example/few-shot-det/code/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--data_root', default=VOCroot, help='Location of VOC root directory')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
# train_sets = 'train'
means = (104, 117, 123)  # only support voc now

accum_batch_size = 32
iter_size = accum_batch_size / args.batch_size
stepvalues = (60000, 80000, 100000)
start_iter = 0
data_split = "fold0_1shot_train"

if args.visdom:
    import visdom
    viz = visdom.Visdom()




class FewShotNet(nn.Module):

    def __init__(self, support_net, det_net):
        super(FewShotNet, self).__init__()
        self.support_net = support_net
        self.det_net = det_net

    def forward(self, first_images, images):
        det_net_result = self.det_net(images)
        return det_net_result




logger.auto_set_dir()

from vgg import vgg16
from pytorchgo.utils.pytorch_utils import model_summary,optimizer_summary


ssd_net = build_ssd('train', args.dim, num_classes)
net = ssd_net
support_net = vgg16(start_channels=6)
fewshotNet = FewShotNet(support_net=support_net, det_net=ssd_net)

model_summary([fewshotNet])


vgg16_state_dict = torch.load("/home/hutao/lab/pytorchgo/example/few-shot-det/code/vgg16-397923af.pth")
print(vgg16_state_dict.keys())
new_params = {}  # model_distill.state_dict().copy()
for i in vgg16_state_dict:
    if "features.0" in i:
        continue
    new_params[i] = vgg16_state_dict[i]
    logger.info("recovering weight for student model(loading vgg16 weight): {}".format(i))

fewshotNet.support_net.load_state_dict(new_params,strict=False)



if args.resume:
    logger.info('Resuming training, loading {}...'.format(args.resume))
    fewshotNet.det_net.load_weights(args.resume)
    start_iter = int(args.resume.split('/')[-1].split('.')[0].split('_')[-1])
else:
    vgg_weights = torch.load(args.basenet)
    logger.info('Loading base network...')
    fewshotNet.det_net.vgg.load_state_dict(vgg_weights)
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
    fewshotNet.det_net.extras.apply(weights_init)
    fewshotNet.det_net.loc.apply(weights_init)
    fewshotNet.det_net.conf.apply(weights_init)

optimizer = optim.SGD(fewshotNet.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, args.dim, 0.5, True, 0, True, 3, 0.5, False, args.cuda)

def DatasetSync(dataset='VOC',split='training'):

    if dataset=='VOC':
        #DataRoot=os.path.join(args.data_root,'VOCdevkit')
        DataRoot=args.data_root
        dataset = VOCDetection(DataRoot, train_sets, SSDAugmentation(
        args.dim, means), AnnotationTransform())
    elif dataset=='kitti':
        DataRoot=os.path.join(args.data_root,'kitti')
        dataset = KittiLoader(DataRoot, split=split,img_size=(1000,300),
                  transforms=SSDAugmentation((1000,300),means),
                  target_transform=AnnotationTransform_kitti())
    elif dataset == "VOC-FEWSHOT":
        from data.FewShotDs import FewShotVOCDataset
        dataset = FewShotVOCDataset(name=data_split)
    return dataset

def train():
    fewshotNet.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    logger.info('Loading Dataset...')

    # dataset = VOCDetection(args.voc_root, train_sets, SSDAugmentation(
    #     args.dim, means), AnnotationTransform())
    dataset=DatasetSync(dataset=args.dataset,split='training')


    epoch_size = len(dataset) // args.batch_size
    logger.info('Training SSD on {}'.format(dataset.name))
    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        epoch_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
    batch_iterator = None
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)

    lr=args.lr
    for iteration in range(start_iter, args.iterations + 1):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)

        if iteration in stepvalues:
            step_index += 1
            lr=adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * epoch,
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                    win=epoch_lot,
                    update='append'
                )
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        first_images, images, targets, metadata = next(batch_iterator)
        #embed()
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        t0 = time.time()
        out = fewshotNet(first_images, images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        if iteration % 10 == 0:
            logger.info('''
                Timer: {:.5f} sec.\t LR: {}.\t Iter: {}.\t Loss_l: {:.5f}.\t Loss_c: {:.5f}.
                '''.format((t1-t0),lr,iteration,loss_l.data[0],loss_c.data[0]))
            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())
        if args.visdom:
            viz.line(
                X=torch.ones((1, 3)).cpu() * iteration,
                Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
                    loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
                win=lot,
                update='append'
            )
            # hacky fencepost solution for 0th epoch plot
            if iteration == 0:
                viz.line(
                    X=torch.zeros((1, 3)).cpu(),
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu(),
                    win=epoch_lot,
                    update=True
                )
        if iteration % 5000 == 0:
            logger.info('Saving state, iter: {}'.format(iteration))
            torch.save(fewshotNet.state_dict(), os.path.join(logger.get_logger_dir(),'cherry-iter{}.pth'.format(iteration)))
    torch.save(fewshotNet.state_dict(), os.path.join(logger.get_logger_dir(),'cherry.pth'))

    
def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5) 
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
