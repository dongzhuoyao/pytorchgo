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

def get_data_path(name):
    """Extract path to data from config file.

    Args:
        name (str): The name of the dataset.

    Returns:
        (str): The path to the root directory containing the dataset.
    """
    js = open('config.json').read()
    data = json.loads(js)
    return os.path.expanduser(data[name]['data_path'])

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
    def __init__(self, root, split='train_aug', is_transform=False,
                 img_size=512, epoch_scale=1, augmentations=None,):
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 21
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) \
                                               else (img_size, img_size)
        self.epoch_scale = epoch_scale
        datalist = "/home/hutao/lab/pytorchgo/dataset_list/pascalvoc12"
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


"""
        im = m.imread(im_path)
        im = np.array(im, dtype=np.uint8)
        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)
        if True:
            pass
        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl


    def transform(self, img, lbl):
        #if  "train" in self.split or "val" :
        img = m.imresize(img, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # HWC -> CHW
        img = img.transpose(2, 0, 1)

        lbl[lbl==255] = 0#The boundary label (255 in ground truth labels) has not been ignored in the loss function in the current version, instead it has been merged with the background.
        # The ignore_label caffe parameter would be implemented in the future versions.
        lbl = lbl.astype(float)
        #if  "train" in self.split:
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest',mode='F')

        lbl = lbl.astype(int)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl
"""




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