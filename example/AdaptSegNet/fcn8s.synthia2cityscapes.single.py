import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
import torchfcn

from model.deeplab_multi import Res_Deeplab
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SynthiaDataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from pytorchgo.utils import logger
from tqdm import tqdm
from pytorchgo.utils.pytorch_utils import model_summary,optimizer_summary

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'FCN8S'
#'DeepLab'
BATCH_SIZE = 2
ITER_SIZE = 1
NUM_WORKERS = 4

IGNORE_LABEL = 255
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19

INPUT_SIZE = '1024,512'  #'1280,720'
INPUT_SIZE_TARGET = '1024,512'


POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0
LAMBDA_ADV_TARGET1 = 0
LAMBDA_ADV_TARGET2 = 0.001

TARGET = 'cityscapes'
SET = 'train'
GPU = 1

#SOURCE_DATA = "GTA5"
SOURCE_DATA = "SYNTHIA"

if SOURCE_DATA == "GTA5":
    DATA_DIRECTORY = './data/GTA5'
    DATA_LIST_PATH = './dataset/gta5_list/train.txt'

    NUM_STEPS = 250000
    NUM_STEPS_STOP = 80000  # early stopping
    SAVE_PRED_EVERY = 2000

elif SOURCE_DATA == "SYNTHIA":
    DATA_DIRECTORY = './data/SYNTHIA'
    DATA_LIST_PATH = './dataset/synthia_list/SYNTHIA_imagelist_train.txt'
    LABEL_LIST_PATH =  './dataset/synthia_list/SYNTHIA_labellist_train.txt'

    NUM_STEPS = 160000
    NUM_STEPS_STOP = 50000  # early stopping
    SAVE_PRED_EVERY = 2000
else:
    raise


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore_label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--input_size_target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is_training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning_rate_D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda_seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda_adv_target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda_adv_target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not_restore_last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random_mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random_scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d().cuda()

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def proceed_test(model, quick_test = 1e10):
    logger.info("proceed test on cityscapes val set...")
    model.eval()
    model.cuda()
    testloader = data.DataLoader(
        cityscapesDataSet(crop_size=(2048, 1024), mean=IMG_MEAN, scale=False, mirror=False, set="val"),
        batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear')

    from tensorpack.utils.stats import MIoUStatistics
    stat = MIoUStatistics(NUM_CLASSES)

    for index, batch in tqdm(enumerate(testloader), desc="validation"):
        if index > quick_test: break

        image, label, _, name = batch
        image, label = Variable(image, volatile=True), Variable(label)

        output1, output2 = model(image.cuda())#(1,19,129,257)
        output = interp(output2).cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        stat.feed(output, label.data.cpu().numpy().squeeze())

    miou16 = np.sum(stat.IoU) / 16
    print("tensorpack class16 IoU: {}".format(miou16))
    model.train()
    return miou16



def main():
    """Create the model and start the training."""

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)

    cudnn.enabled = True
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Create network
    if args.model == 'DeepLab':
        logger.info("adopting Deeplabv2 base model..")
        model = Res_Deeplab(num_classes=args.num_classes, multi_scale = False)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)

        optimizer = optim.SGD(model.optim_parameters(args),
                              lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.model == "FCN8S":
        logger.info("adopting FCN8S base model..")
        from pytorchgo.model.MyFCN8s import MyFCN8s
        model  = MyFCN8s(n_class=NUM_CLASSES)
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)

        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    else:
        raise

    model.train()
    model.cuda()

    cudnn.benchmark = True

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D2 = FCDiscriminator(num_classes=args.num_classes)



    model_D1.train()
    model_D1.cuda()

    model_D2.train()
    model_D2.cuda()

    if SOURCE_DATA == "GTA5":
        trainloader = data.DataLoader(
            GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                        crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        trainloader_iter = enumerate(trainloader)
    elif SOURCE_DATA == "SYNTHIA":
        trainloader = data.DataLoader(
            SynthiaDataSet(args.data_dir, args.data_list,LABEL_LIST_PATH, max_iters=args.num_steps * args.iter_size * args.batch_size,
                        crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        trainloader_iter = enumerate(trainloader)
    else:
        raise



    targetloader = data.DataLoader(cityscapesDataSet(
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)


    targetloader_iter = enumerate(targetloader)

    # implement model.optim_parameters(args) to handle different models' lr setting


    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')

    # labels for adversarial training
    source_label = 0
    target_label = 1

    best_mIoU = 0

    model_summary([model,model_D1,model_D2])
    optimizer_summary([optimizer,optimizer_D1,optimizer_D2])

    for i_iter in tqdm(range(args.num_steps_stop),total=args.num_steps_stop, desc="training"):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        lr_D1 = adjust_learning_rate_D(optimizer_D1, i_iter)
        lr_D2 = adjust_learning_rate_D(optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):

            ######################### train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source

            _, batch = trainloader_iter.next()
            images, labels, _, _ = batch
            images = Variable(images).cuda()

            pred2 = model(images)
            pred2 = interp(pred2)

            loss_seg2 = loss_calc(pred2, labels)
            loss = loss_seg2

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value2 += loss_seg2.data.cpu().numpy()[0] / args.iter_size

            # train with target

            _, batch = targetloader_iter.next()
            images, _, _ ,_ = batch
            images = Variable(images).cuda()

            pred_target2 = model(images)
            pred_target2 = interp_target(pred_target2)

            D_out2 = model_D2(F.softmax(pred_target2))


            loss_adv_target2 = bce_loss(D_out2,
                                        Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(
                                            ))

            loss = args.lambda_adv_target2 * loss_adv_target2
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy()[0] / args.iter_size

            ################################## train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred2 = pred2.detach()
            D_out2 = model_D2(F.softmax(pred2))


            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())

            loss_D2 = loss_D2 / args.iter_size / 2
            loss_D2.backward()

            loss_D_value2 += loss_D2.data.cpu().numpy()[0]

            # train with target
            pred_target2 = pred_target2.detach()

            D_out2 = model_D2(F.softmax(pred_target2))

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda())

            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D2.backward()

            loss_D_value2 += loss_D2.data.cpu().numpy()[0]

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()

        if i_iter%100 ==0:
            logger.info(
        'iter = {}/{},loss_seg1 = {:.3f} loss_seg2 = {:.3f} loss_adv1 = {:.3f}, loss_adv2 = {:.3f} loss_D1 = {:.3f} loss_D2 = {:.3f}, lr={:.7f}, lr_D={:.7f}, best miou16= {:.5f}'.format(
            i_iter, args.num_steps_stop, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2, lr, lr_D1, best_mIoU))



        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            logger.info("saving snapshot.....")
            cur_miou16 = proceed_test(model)
            is_best = True if best_mIoU < cur_miou16 else False
            if is_best:
                best_mIoU = cur_miou16
            torch.save({
                'iteration': i_iter,
                'optim_state_dict': optimizer.state_dict(),
                'optim_D1_state_dict': optimizer_D1.state_dict(),
                'optim_D2_state_dict': optimizer_D2.state_dict(),
                'model_state_dict': model.state_dict(),
                'model_D1_state_dict': model_D1.state_dict(),
                'model_D2_state_dict': model_D2.state_dict(),
                'best_mean_iu': cur_miou16,
            }, osp.join(logger.get_logger_dir(), 'checkpoint.pth.tar'))
            if is_best:
                import shutil
                shutil.copy(osp.join(logger.get_logger_dir(), 'checkpoint.pth.tar'),
                            osp.join(logger.get_logger_dir(), 'model_best.pth.tar'))


        if i_iter >= args.num_steps_stop - 1:
            break



if __name__ == '__main__':
    logger.auto_set_dir()
    main()
