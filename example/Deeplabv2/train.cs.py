import torch
import argparse
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
from model import Res_Deeplab
from loss import CrossEntropy2d
from datasets import VOCDataSet,CSDataSet
import random
import timeit
from tqdm import tqdm
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 3

IGNORE_LABEL = 255
INPUT_SIZE = '672,672'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 20
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'resnet101-5d3b4d8f.pth' #'http://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
NUM_STEPS = 40000
SAVE_PRED_EVERY = 3000
WEIGHT_DECAY = 0.0005

is_debug = 0


if is_debug == 1:
    SAVE_PRED_EVERY = 5
    NUM_STEPS = 6

from pytorchgo.utils import logger


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random_scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()

args = get_arguments()

logger.auto_set_dir()

random.seed(args.random_seed)

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL).cuda()
    
    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)

    
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i
            
            
def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def main():
    """Create the model and start the training."""
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    # Create network.
    model = Res_Deeplab(num_classes=args.num_classes,aspp=False)
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    saved_state_dict = torch.load(args.restore_from)
    new_params = {}#model.state_dict().copy()
    for i in saved_state_dict:
        #Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        # print i_parts
        if not args.num_classes == 21 or not i_parts[1]=='layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            logger.info("recovering weight: {}".format(i))
    model.load_state_dict(new_params,strict=False)
    #model.float()
    #model.eval() # use_global_stats = True
    model.train()
    model.cuda()
    
    cudnn.benchmark = True



    from pytorchgo.augmentation.segmentation import SubtractMeans, PIL2NP, RGB2BGR, PIL_Scale, Value255to0, ToLabel, \
        PascalPadding,RandomCrop
    from torchvision.transforms import Compose, Normalize, ToTensor

    img_transform = Compose([  # notice the order!!!
        SubtractMeans(),
        # RandomScale()
    ])

    label_transform = Compose([
    ])

    augmentation = Compose(
        [
            RandomCrop(input_size)
            #PascalPadding(input_size), cityscapes image is very large!
        ]
    )

    trainloader = data.DataLoader(CSDataSet(max_iters=args.num_steps*args.batch_size,
                     mirror=args.random_mirror, img_transform=img_transform, label_transform=label_transform, augmentation=augmentation),
                    batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)

    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.learning_rate }, 
                {'params': get_10x_lr_params(model), 'lr': 10*args.learning_rate}], 
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()


    from pytorchgo.augmentation.segmentation import SubtractMeans, PIL2NP, RGB2BGR, PIL_Scale, Value255to0, ToLabel, RandomCrop, \
        PascalPadding
    from torchvision.transforms import Compose, Normalize, ToTensor

    val_img_transform = Compose([  # notice the order!!!
        SubtractMeans(),
        # RandomScale()
    ])

    val_label_transform = Compose([
        # PIL_Scale(train_img_shape, Image.NEAREST),
        PIL2NP(),
        Value255to0(),
        ToLabel()

    ])

    val_augmentation = Compose(
        [
        #PascalPadding((1024,2048))
        ]
    )

    testloader = data.DataLoader(dataset=CSDataSet(name="val", img_transform=val_img_transform, label_transform=val_label_transform, augmentation=val_augmentation), batch_size=1, shuffle=False,pin_memory=True)

    interp = nn.Upsample(size=input_size, mode='bilinear')
    data_list = []

    from pytorchgo.utils.pytorch_utils import model_summary, optimizer_summary

    model_summary(model)
    optimizer_summary(optimizer)

    best_miou = 0

    for i_iter, batch in tqdm(enumerate(trainloader), total=len(trainloader), desc="training deeplab"):
        images, labels, _, _ = batch
        images = Variable(images).cuda()

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        pred = interp(model(images))
        loss = loss_calc(pred, labels)
        loss.backward()
        optimizer.step()

        if i_iter % 50 == 0:
            logger.info('loss = {}, lr={}, best_miou={}'.format(loss.data.cpu().numpy(), lr, best_miou))


        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            logger.info('validation...')
            from evaluate import do_eval_dataset
            model.eval()
            ious = do_eval_dataset(model=model, testloader=testloader, num_classes=NUM_CLASSES, output_size=input_size, quick_eval=100)
            cur_miou = np.mean(ious[1:])
            model.train()

            is_best = True if cur_miou > best_miou else False
            if is_best:
                best_miou = cur_miou
                logger.info('taking snapshot...')
                torch.save({
                    'iteration': i_iter,
                    'optim_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'best_mean_iu': best_miou,
                }, osp.join(logger.get_logger_dir(), 'love.pth'))
            else:
                logger.info("current snapshot is not good enough, skip~~")
            if is_debug==1:
                logger.info("debug mode, break...")
                break


    logger.info('final validation...')
    from evaluate import do_eval_dataset
    model.eval()
    do_eval_dataset(model=model, testloader=testloader, num_classes=NUM_CLASSES, output_size=input_size, restore_from=osp.join(logger.get_logger_dir(), 'love.pth'))
    logger.info("Congrats~")


if __name__ == '__main__':
    main()
