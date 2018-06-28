import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import v as cfg
import os
from IPython import embed


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(cfg[str(size)])
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        from vgg_fcn import vgg16
        self.support_net = vgg16(start_channels=3)


        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, self.size, bkg_label=0, top_k=200, conf_thresh=0.01, nms_thresh=0.45)

    def forward(self, support_images, x, is_train):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300]. or [batch,3,512,512]

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch,num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors,4]
                    3: priorbox layers, Shape: [num_priors,4]
        """
        sources = list()
        loc = list()
        conf = list()

        support_result = self.support_net(support_images)


        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            if k == 30:
                x = self.vgg[k](x) + support_result#support !!!
            else:
                x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if not is_train:#not is_train.data.numpy()[0]:
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, size, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    # SSD512 need add one more Conv layer(Conv12_2)
    if size == 512:
        layers += [nn.Conv2d(in_channels, 256, kernel_size=4, padding=1)]
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [24, -2]
    for k, v in enumerate(vgg_source):
        try:
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                            cfg[k] * num_classes, kernel_size=3, padding=1)]
        except:
            import ipdb
            ipdb.set_trace()
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 6, 4, 4],
}


def build_ssd(size=512, num_classes=21):
    if size != 300 and size != 512:
        print("Error: Sorry only SSD300 or SSD512 is supported currently!")
        return

    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
             add_extras(extras[str(size)], size, 1024),
             mbox[str(size)], num_classes)

    return SSD(size, base_, extras_, head_,  num_classes)











############################################################################################################################################################################################


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

#from data import detection_collate
from data.FewShotDs import detection_collate

from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
import numpy as np
import time
from tqdm import tqdm
from pytorchgo.utils import logger
from data.FewShotDs import FewShotVOCDataset
from pytorchgo.utils.pytorch_utils import model_summary, optimizer_summary

is_debug = 1
num_classes = 2
iterations = 120000
stepvalues = (60000, 80000, 100000)
start_iter = 0
log_per_iter = 500
save_per_iter = 5000
image_size = 300
train_data_split = "fold0_1shot_train"
val_data_split = "fold0_1shot_val"
gpu = '4'
quick_eval = 1e10

if is_debug == 1:
    log_per_iter = 10
    save_per_iter = 100
    quick_eval = 400


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--dim', default=image_size, type=int, help='Size of the input image, only support 300 or 512')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=iterations, type=int, help='Number of training iterations')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--gpu', default=gpu)
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from pytorchgo.utils.pytorch_utils import  set_gpu
set_gpu(args.gpu)
torch.cuda.set_device(int(args.gpu))


if args.visdom:
    import visdom
    viz = visdom.Visdom()


def train():

    logger.info("current cuda device: {}".format(torch.cuda.current_device()))

    few_shot_net = build_ssd(args.dim, num_classes)

    vgg16_state_dict = torch.load("vgg16-397923af.pth")
    new_params = {}
    for i in vgg16_state_dict:
        if "features.0" in i:
            continue
        new_params[i] = vgg16_state_dict[i]
        logger.info("recovering weight for student model(loading vgg16 weight): {}".format(i))
    few_shot_net.support_net.load_state_dict(new_params, strict=False)


    logger.info('Loading base network...')
    few_shot_net.vgg.load_state_dict(torch.load(args.basenet))
    start_iter = 0

    if args.cuda:
        few_shot_net = few_shot_net.cuda()

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            m.bias.data.zero_()

    logger.info('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    few_shot_net.extras.apply(weights_init)
    few_shot_net.loc.apply(weights_init)
    few_shot_net.conf.apply(weights_init)

    optimizer = optim.SGD(few_shot_net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(num_classes, size=args.dim, overlap_thresh=0.5, prior_for_matching=True, bkg_label=0, neg_mining=True, neg_pos=3, neg_overlap=0.5, encode_target=False, use_gpu=args.cuda)

    model_summary(few_shot_net)
    optimizer_summary(optimizer)




    few_shot_net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    best_result = 0
    logger.info('Loading Dataset...')



    dataset = FewShotVOCDataset(name=train_data_split, image_size=(args.dim, args.dim))

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
                                  shuffle=True, pin_memory=True, collate_fn=detection_collate)

    lr=args.lr
    for iteration in tqdm(range(start_iter, args.iterations + 1),total=args.iterations, desc="training {}".format(logger.get_logger_dir())):
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
            first_images = Variable(first_images.cuda())
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            first_images = Variable(first_images)
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        out = few_shot_net(first_images, images, is_train =True)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        if iteration % log_per_iter == 0 and iteration>0:
            logger.info('''LR: {}\t Iter: {}\t Loss_l: {:.5f}\t Loss_c: {:.5f}\t Loss_total: {:.5f}\t best_result: {:.5f}'''.format(lr,iteration,loss_l.data[0],loss_c.data[0], loss.data[0], best_result))
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
        if iteration % save_per_iter == 0 and iteration>0:
            few_shot_net.eval()
            cur_eval_result = do_eval(few_shot_net)
            few_shot_net.train()

            is_best = True if cur_eval_result > best_result else False
            if is_best:
                best_result = cur_eval_result
                torch.save({
                    'iteration': iteration,
                    'optim_state_dict': optimizer.state_dict(),
                    'model_state_dict': few_shot_net.state_dict(),
                    'best_mean_iu': best_result,
                }, os.path.join(logger.get_logger_dir(), 'cherry.pth'))
            else:
                logger.info("current snapshot is not good enough, skip~~")

            logger.info('current iter: {} current_result: {:.5f}'.format(iteration, cur_eval_result))

    logger.info("Congrats~")



def do_eval(few_shot_net):


    tmp_eval = os.path.join(logger.get_logger_dir(), "eval_tmp")

    if os.path.isdir(tmp_eval):
        import shutil
        shutil.rmtree(tmp_eval)
    os.makedirs(tmp_eval)

    ground_truth_dir = os.path.join(tmp_eval, "ground-truth")
    predicted_dir = os.path.join(tmp_eval, "predicted")
    os.makedirs(ground_truth_dir)
    os.makedirs(predicted_dir)




    dataset = FewShotVOCDataset(name=val_data_split)
    num_images = len(dataset)

    data_loader = data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers,
                                  shuffle=True, pin_memory=True, collate_fn=detection_collate)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    w = image_size
    h = image_size

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="online evaluation"):
        if i > quick_eval:break
        with open(os.path.join(ground_truth_dir, "{}.txt".format(i)), "w") as f_gt:
            with open(os.path.join(predicted_dir, "{}.txt".format(i)), "w") as f_predict:
                #if i > 500:break
                first_images, images, targets, metadata = batch

                if args.cuda:
                    first_images = Variable(first_images.cuda())
                    x = Variable(images.cuda())
                    #x = Variable(images.unsqueeze(0).cuda())
                else:
                    first_images = Variable(first_images)
                    x = Variable(images)

                gt_bboxes = targets[0].numpy()
                for _ in range(gt_bboxes.shape[0]):
                    gt_bboxes[_, 0] *= w
                    gt_bboxes[_, 2] *= w
                    gt_bboxes[_, 1] *= h
                    gt_bboxes[_, 3] *= h
                    f_gt.write("shit {} {} {} {}\n".format(int(gt_bboxes[_, 0]),int(gt_bboxes[_, 1]),int(gt_bboxes[_, 2]),int(gt_bboxes[_, 3])))


                detections = few_shot_net(first_images, x, is_train = False).data


                # skip j = 0, because it's the background class
                for j in range(1, detections.size(1)):
                    dets = detections[0, j, :]
                    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                    dets = torch.masked_select(dets, mask).view(-1, 5)
                    if dets.dim() == 0:
                        continue
                    boxes = dets[:, 1:]
                    boxes[:, 0] *= w
                    boxes[:, 2] *= w
                    boxes[:, 1] *= h
                    boxes[:, 3] *= h
                    scores = dets[:, 0].cpu().numpy()
                    cls_dets = np.hstack((boxes.cpu().numpy(),
                                          scores[:, np.newaxis])).astype(np.float32,
                                                                         copy=False)
                    all_boxes[j][i] = cls_dets

                    for _ in range(cls_dets.shape[0]):
                        f_predict.write("shit 1 {} {} {} {}\n".format(cls_dets[_, 0], cls_dets[_, 1], cls_dets[_, 2], cls_dets[_, 3]))



    from eval_map import eval_online
    mAP = eval_online(tmp_eval)
    return mAP




    
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
    logger.auto_set_dir()
    train()
