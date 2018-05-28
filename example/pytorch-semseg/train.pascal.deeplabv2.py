#Notice for deeplabv2
# Image scale to [-127, 128]

import sys, os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *
from tqdm import tqdm
from pytorchgo.utils import logger
from pytorchgo.utils.pytorch_utils import optimizer_summary
from pytorchgo.utils.learning_rate import adjust_learning_rate

from pytorchgo.loss import CrossEntropyLoss2d_Seg

is_debug = 0

def train(args):

    logger.auto_set_dir()
    from pytorchgo.utils.pytorch_utils import set_gpu
    set_gpu(2)

    # Setup Augmentations
    data_aug= Compose([RandomRotate(10),                                        
                       RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), epoch_scale=1, augmentations=data_aug, img_norm=False)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)


    # Setup Metrics
    running_metrics = runningScore(n_classes)


    # Setup Model
    from pytorchgo.model.deeplabv1 import VGG16_LargeFoV
    from pytorchgo.model.deeplab_resnet import Res_Deeplab

    model = Res_Deeplab(NoLabels=n_classes, pretrained=True)

    from pytorchgo.utils.pytorch_utils import model_summary,optimizer_summary
    model_summary(model)




    def get_validation_miou(model):
        model.eval()
        v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(513, 513), img_norm=False)
        valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader), total=len(valloader), desc="validation"):
            if i_val > 5 and is_debug: break
            img_large = torch.Tensor(np.zeros((1, 3, 513, 513)))
            img_large[:, :, :images_val.shape[2], :images_val.shape[3]] = images_val

            output = model(Variable(img_large, volatile=True).cuda())
            output = output[0]
            output = output.data.max(1)[1].cpu().numpy()
            pred = output[:, :images_val.shape[2], :images_val.shape[3]]

            gt = labels_val.numpy()

            running_metrics.update(gt, pred)

        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            logger.info("{}: {}".format(k, v))
        running_metrics.reset()
        return score['Mean IoU : \t']


    model.cuda()
    
    # Check if model has custom optimizer / loss
    if hasattr(model, 'optimizer'):
        logger.warn("don't have customzed optimizer, use default setting!")
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.optimizer_params(args.l_rate), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    optimizer_summary(optimizer)
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            logger.info("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            logger.info("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("No checkpoint found at '{}'".format(args.resume))

    best_iou = 0
    logger.info('start!!')
    for epoch in tqdm(range(args.n_epoch),total=args.n_epoch):
        model.train()
        for i, (images, labels) in tqdm(enumerate(trainloader),total=len(trainloader), desc="training epoch {}/{}".format(epoch, args.n_epoch)):
            if i > 10 and is_debug: break

            if i> 200:break

            cur_iter = i + epoch*len(trainloader)
            cur_lr = adjust_learning_rate(optimizer,args.l_rate,cur_iter,args.n_epoch*len(trainloader),power=0.9)


            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images) # use fusion score
            loss = CrossEntropyLoss2d_Seg(input=outputs[0], target=labels, class_num=n_classes)

            #for i in range(len(outputs) - 1):
            #for i in range(1):
            #    loss = loss + CrossEntropyLoss2d_Seg(input=outputs[i], target=labels, class_num=n_classes)

            loss.backward()
            optimizer.step()


            if (i+1) % 100 == 0:
                logger.info("Epoch [%d/%d] Loss: %.4f, lr: %.7f, best mIoU: %.7f" % (epoch+1, args.n_epoch, loss.data[0], cur_lr, best_iou))


        cur_miou = get_validation_miou(model)

        if cur_miou >= best_iou:
            best_iou = cur_miou
            state = {'epoch': epoch+1,
                     'mIoU': best_iou,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, os.path.join(logger.get_logger_dir(), "best_model.pth"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='deeplabv1',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=513,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=513,
                        help='Width of the input image')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=False)

    parser.add_argument('--n_epoch', nargs='?', type=int, default=16,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=2.5e-4, # original implementation of deeplabv1 learning rate is 1e-3 and poly update
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')


    args = parser.parse_args()
    train(args)
