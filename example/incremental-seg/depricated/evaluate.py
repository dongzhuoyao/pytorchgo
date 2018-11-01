import torch
import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys

from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
from model import Res_Deeplab
from datasets import VOCDataSet
from collections import OrderedDict
import os
from tqdm import tqdm

import torch.nn as nn
from pytorchgo.utils import  logger

DATA_DIRECTORY = '/data1/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012'
DATA_LIST_PATH = 'datalist/val.txt'
NUM_CLASSES = 21
NUM_STEPS = 1449 # Number of images in the validation set.
input_size = (513, 513)
RESTORE_FROM = '/data4/hutao/pytorchgo/example/incremental-seg/train_log/train.473/VOC12_scenes_20000.pth'#'/home/hutao/lab/Pytorch-Deeplab/VOC12_scenes_20000.pth'

#python evaluate.py --num_classes 20 --restore_from train_log/train.473.class19meaning/VOC12_scenes_20000.pth
def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=2,
                        help="choose gpu device.")
    return parser.parse_args()



def get_iou(data_list, class_num, ):
    from multiprocessing import Pool 
    from metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    logger.info('meanIOU-with bg: {}'.format(str(aveJ)))
    logger.info('meanIOU-without bg: {}'.format(np.mean(j_list[1:])))
    logger.info("IOU: {}".format(str(j_list)))
    logger.info("confusion matrix: {}".format(M))

    return j_list



def show_all(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5), 
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5), 
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0), 
                    (0.5,0.75,0),(0,0.25,0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()



def do_eval(model, data_dir, data_list, num_classes, restore_from = None):

    if restore_from is not None:
        saved_state_dict = torch.load(restore_from)
        model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    from pytorchgo.augmentation.segmentation import SubtractMeans
    from PIL import Image

    from pytorchgo.augmentation.segmentation import SubtractMeans, PIL2NP, RGB2BGR, PIL_Scale, Value255to0, ToLabel, \
        PascalPadding
    from torchvision.transforms import Compose, Normalize, ToTensor

    img_transform = Compose([  # notice the order!!!
        SubtractMeans(),
        # RandomScale()
    ])

    label_transform = Compose([
        # PIL_Scale(train_img_shape, Image.NEAREST),
        PIL2NP(),
        Value255to0(),
        ToLabel()

    ])

    augmentation = Compose(
        [
            PascalPadding(input_size)
        ]
    )

    testloader = data.DataLoader(
        VOCDataSet(data_dir, data_list, mirror=False, img_transform=img_transform, augmentation=augmentation),
        batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=input_size, mode='bilinear')
    interp = nn.Upsample(size=input_size, mode='bilinear')
    data_list = []

    for index, batch in tqdm(enumerate(testloader)):
        #if index > 10: break
        image, label, size, name = batch
        size = size[0].numpy()
        output = model(Variable(image, volatile=True).cuda())
        output = interp(output).cpu().data[0].numpy()

        output = output[:, :size[0], :size[1]]
        gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=np.int)

        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

        # show_all(gt, output)
        data_list.append([gt.flatten(), output.flatten()])

    return get_iou(data_list, num_classes)


def main():
    args = get_arguments()
    gpu0 = args.gpu
    from pytorchgo.utils.pytorch_utils import set_gpu
    #set_gpu(gpu0)
    #torch.cuda.set_device(int(gpu0))
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5'

    model = Res_Deeplab(num_classes=args.num_classes)
    
    do_eval(model, restore_from=args.restore_from, data_dir=args.data_dir, data_list = args.data_list, num_classes = args.num_classes)



if __name__ == '__main__':
    main()