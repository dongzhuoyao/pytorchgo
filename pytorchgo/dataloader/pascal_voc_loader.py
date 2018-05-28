import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob

from tqdm import tqdm
from torch.utils import data
from PIL import Image


class pascalVOCLoader(data.Dataset):
    """Data loader for the Pascal VOC semantic segmentation dataset.

    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset
    (where annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.

    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.

    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        trainval: The combination of `train` and `val` - 2913 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_aug_val: The original VOC 2012 validation data minus the images
                   present in `train_aug` (This is done with the same logic as
                   the validation set used in FCN PAMI paper, but with VOC 2012
                   rather than VOC 2011) - 904 images
    """
    def __init__(self, split='train_aug',
                  epoch_scale=1,img_transform=None, label_transform=None):
        assert  split in ['train', 'train_aug', 'val']
        self.root = '/home/hutao/dataset/pascalvoc2012'
        datalist = "/home/hutao/lab/pytorchgo/dataset_list/pascalvoc12"
        self.split = split
        self.n_classes = 21
        self.label_transform = label_transform
        self.img_transform = img_transform
        self.files = collections.defaultdict(list)
        self.epoch_scale = epoch_scale
        if self.split == 'train':
            with open(os.path.join(datalist, 'train.txt'),'r') as f:
                lines = f.readlines()
                self.files[self.split] = [tmp.strip() for tmp in lines] * epoch_scale
                # here multiply a epoch scale to make the epoch size scalable!!

        if self.split == 'train_aug':
            with open(os.path.join(datalist, 'train_aug.txt'),'r') as f:
                lines = f.readlines()
                self.files[self.split] = [tmp.strip() for tmp in lines]

        if self.split == 'val':
            with open(os.path.join(datalist, 'val.txt'),'r') as f:
                lines = f.readlines()
                self.files[self.split] = [tmp.strip() for tmp in lines]






    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_label_str = self.files[self.split][index]
        im_path, lbl_path = img_label_str.strip().split()
        im_path = os.path.join(self.root, 'VOC2012trainval/VOCdevkit/VOC2012', im_path)
        lbl_path = os.path.join(self.root,  'VOC2012trainval/VOCdevkit/VOC2012', lbl_path)

        img_file = im_path
        img = Image.open(img_file).convert('RGB')
        label_file = lbl_path
        label = Image.open(label_file).convert("P")

        if False:
            print img_file
            print label_file

        if self.img_transform:
            img = self.img_transform(img)
        if self.label_transform:
            label = self.label_transform(label)

        return img, label



if __name__ == '__main__':


    t_loader = pascalVOCLoader('~/dataset/pascalvoc2012', split='val', is_transform=True, img_size=(513, 513), epoch_scale=1, augmentations=None, img_norm=False)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=1, num_workers=1, shuffle=True)
    for idx, data in enumerate(trainloader):
        print idx
        image, label = data
        #image = image.numpy()
        #label = label.numpy()
        from pytorchgo.utils.vis import vis_seg
        vis_seg(image, label)