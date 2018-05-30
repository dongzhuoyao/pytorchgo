import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab_multi import Res_Deeplab
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result/cityscapes'

IGNORE_LABEL = 255
NUM_CLASSES = 19
is_debug = 1
#RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM = 'train_log/deeplabv2.synthia2cityscapes.single.8k/model_best.pth'



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default='val',
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""
    import warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    args = get_arguments()


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = Res_Deeplab(num_classes=args.num_classes)
    #from pytorchgo.model.MyFCN8s import MyFCN8s
    #model = MyFCN8s(n_class=NUM_CLASSES)

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict['model_state_dict'])

    model.eval()
    model.cuda()

    image_size = (1024, 512)#(1280,720) #(2048, 1024)
    cityscape_image_size = (2048, 1024)

    print ("evaluating {}".format(args.restore_from))
    print ("************ best mIoU:{} *******".format(saved_state_dict['best_mean_iu']))
    print("evaluation image size: {}, please make sure this image size is equal to your training image size, this is important for your final mIoU!".format(image_size))

    testloader = data.DataLoader(cityscapesDataSet( crop_size=(image_size[0], image_size[1]), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(cityscape_image_size[1], cityscape_image_size[0]), mode='bilinear')

    from tensorpack.utils.stats import MIoUStatistics
    stat = MIoUStatistics(NUM_CLASSES)

    for index, batch in tqdm(enumerate(testloader)):
        if index > 10: break
        image,label, _, name = batch
        if index == 0 and is_debug == 1:
            pass
            print name

        image, label = Variable(image, volatile=True), Variable(label)

        output1, output2 = model(image.cuda())
        output = interp(output2).cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        stat.feed(output, label.data.cpu().numpy().squeeze())

    print("tensorpack IoU: {}".format(stat.mIoU_beautify))
    print("tensorpack class16 IoU: {}".format(np.sum(stat.IoU)/16))
    print("tensorpack mIoU: {}".format(stat.mIoU))
    print("tensorpack mean_accuracy: {}".format(stat.mean_accuracy))
    print("tensorpack accuracy: {}".format(stat.accuracy))


if __name__ == '__main__':
    main()
