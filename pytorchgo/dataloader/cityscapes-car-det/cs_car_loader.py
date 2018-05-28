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
from PIL import Image, ImageFont, ImageDraw, ImageEnhance


class CsCarLoader(data.Dataset):
    def __init__(self, split='train',
                  epoch_scale=1,img_transform=None, label_transform=None):
        assert  split in ['train', 'val']
        self.root = '/data1/dataset/cityscapes'
        datalist = "/home/hutao/lab/pytorchgo/dataset_list/cityscapes_car"
        self.split = split
        self.n_classes = 21
        self.label_transform = label_transform
        self.img_transform = img_transform
        self.files = collections.defaultdict(list)
        self.epoch_scale = epoch_scale
        if self.split == 'train':
            with open(os.path.join(datalist, 'car_train.txt'),'r') as f:
                lines = f.readlines()
                self.files[self.split] = [tmp.strip() for tmp in lines] * epoch_scale
                # here multiply a epoch scale to make the epoch size scalable!!


        if self.split == 'val':
            with open(os.path.join(datalist, 'car_val.txt'),'r') as f:
                lines = f.readlines()
                self.files[self.split] = [tmp.strip() for tmp in lines]






    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_label_str = self.files[self.split][index]
        if self.split == 'train':
            im_path = img_label_str
            im_path = os.path.join(self.root, 'leftImg8bit/train', im_path)

            img_file = im_path
            img = Image.open(img_file).convert('RGB')

            if False:
                print img_file

            if self.img_transform:
                img = self.img_transform(img)

            return img
        elif self.split == "val":
            im_path, boundbox_liststr = img_label_str.strip().split(" ")
            boundbox_list = [[tmp.split(",")[0],tmp.split(",")[1],tmp.split(",")[2],tmp.split(",")[3]] for tmp in boundbox_liststr.split(";")]
            im_path = os.path.join(self.root, 'leftImg8bit/val', im_path)

            img_file = im_path
            img = Image.open(img_file).convert('RGB')


            if True:
                draw = ImageDraw.Draw(img)
                for bb in boundbox_list:
                    min_x, max_x, min_y, max_y = bb
                    draw.rectangle(((int(min_x), int(min_y)), (int(max_x), int(max_y))), outline="red")


                img.save("vis.jpg", "JPEG")

            if self.img_transform:
                img = self.img_transform(img)

            return img
        else:
            raise ValueError



if __name__ == '__main__':

    from pytorchgo.augmentation.segmentation import PIL2NP
    from torchvision.transforms import Compose, Normalize, ToTensor
    t_loader = CsCarLoader( split='val', img_transform=Compose([PIL2NP()]))

    trainloader = data.DataLoader(t_loader, batch_size=1, num_workers=1, shuffle=True)
    for idx, data in enumerate(trainloader):
        print idx
        #image, label = data
        #image = image.numpy()
        #label = label.numpy()
        #from pytorchgo.utils.vis import vis_seg
        #vis_seg(image, label)