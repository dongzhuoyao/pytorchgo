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
from datasets import VOCDataSet
import random
import timeit
from tqdm import tqdm
start = timeit.default_timer()


BATCH_SIZE = 9
DATA_DIRECTORY = '/home/hutao/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012'
DATA_LIST_PATH = 'datalist/train_aug.txt'
IGNORE_LABEL = 255
INPUT_SIZE = (473,473)
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 20000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'resnet50-19c8e357.pth' #'http://download.pytorch.org/models/resnet50-19c8e357.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
WEIGHT_DECAY = 0.0005

from pytorchgo.utils import logger


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input_size",  default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
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
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=2,
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


def main():
    """Create the model and start the training."""
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    input_size = INPUT_SIZE

    cudnn.enabled = True

    # Create network.
    model = Res_Deeplab(num_classes=args.num_classes)
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    from pytorchgo.utils.pytorch_utils import model_summary, optimizer_summary

    model_summary(model)
    saved_state_dict = torch.load(args.restore_from)
    print(saved_state_dict.keys())
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        #Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        # print i_parts
        if  i_parts[0]=='layer5' or i_parts[0]=='fc':
            continue
        new_params[i] = saved_state_dict[i]
        logger.info("recovering weight: {}".format(i))

    model.load_state_dict(new_params)
    #model.float()
    #model.eval() # use_global_stats = True
    model.train()
    model.cuda()
    
    cudnn.benchmark = True



    from pytorchgo.augmentation.segmentation import SubtractMeans, PIL2NP, RGB2BGR, PIL_Scale, Value255to0, ToLabel, \
        PascalPadding
    from torchvision.transforms import Compose, Normalize, ToTensor

    img_transform = Compose([  # notice the order!!!
        SubtractMeans(),
        # RandomScale()
    ])

    label_transform = Compose([
    ])

    augmentation = Compose(
        [
            PascalPadding(input_size)
        ]
    )


    trainloader = data.DataLoader(VOCDataSet(args.data_dir, args.data_list, max_iters=args.num_steps*args.batch_size,
                     mirror=args.random_mirror, img_transform=img_transform, label_transform=label_transform, augmentation=augmentation),
                    batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)

    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.learning_rate }, 
                {'params': get_10x_lr_params(model), 'lr': 10*args.learning_rate}], 
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_summary(optimizer)

    interp = nn.Upsample(size=input_size, mode='bilinear')


    for i_iter, batch in tqdm(enumerate(trainloader), total=len(trainloader), desc="training deeplab"):

        #if i_iter > 10:break

        images, labels, _, _ = batch
        images = Variable(images).cuda()

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        pred = interp(model(images))
        loss = loss_calc(pred, labels)
        loss.backward()
        optimizer.step()

        
        logger.info('loss = {}'.format(loss.data.cpu().numpy()))

        if i_iter >= args.num_steps-1:
            logger.info( 'save model ...')
            torch.save(model.state_dict(),osp.join(logger.get_logger_dir(), 'VOC12_scenes_'+str(args.num_steps)+'.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            logger.info('taking snapshot ...')
            torch.save(model.state_dict(),osp.join(logger.get_logger_dir(), 'VOC12_scenes_'+str(i_iter)+'.pth'))

    logger.info("training done, start evaluation")
    from evaluate import do_eval
    model.eval()
    do_eval(model=model, data_dir= args.data_dir, data_list= args.data_list, num_classes=args.num_classes)
    logger.info("Congrats~")



if __name__ == '__main__':
    main()
