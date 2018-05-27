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
#RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM = 'train_log/deeplabv2.synthia2cityscapes.single/model_best.pth.tar'
SET = 'val'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

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
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
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

    gpu0 = args.gpu

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
    model.cuda(gpu0)

    image_size = (1280,720) #(2048, 1024)
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
        image,label, _, name = batch
        image, label = Variable(image, volatile=True), Variable(label)

        #output2 = model(image.cuda(gpu0))
        output1, output2 = model(image.cuda(gpu0))
        output = interp(output2).cpu().data[0].numpy()

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)


        name = name[0].split('/')[-1]
        output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))

        stat.feed(output, label.data.cpu().numpy().squeeze())

    print("tensorpack class16 IoU: {}".format(np.sum(stat.IoU)/16))
    print("tensorpack mIoU: {}".format(stat.mIoU))
    print("tensorpack mean_accuracy: {}".format(stat.mean_accuracy))
    print("tensorpack accuracy: {}".format(stat.accuracy))


if __name__ == '__main__':
    main()
