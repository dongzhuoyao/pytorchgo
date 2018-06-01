import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import glob

from tqdm import tqdm
from torch.utils import data
from PIL import Image
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

CsCar_CLASSES = (  # always index 0
    'car',)


class CsCarDetection(data.Dataset):
    def __init__(self, split='train',
                  epoch_scale=1,transform=None):
        assert  split in ['train', 'val']
        self.root = '/home/hutao/dataset/cityscapes'
        self.transform = transform
        self.name = "CsCar"
        datalist = "/home/hutao/lab/pytorchgo/dataset_list/cityscapes_car"
        self.split = split
        self.n_classes = 1

        self.files = []
        self.epoch_scale = epoch_scale
        if self.split == 'train':
            with open(os.path.join(datalist, 'car_train.txt'),'r') as f:
                lines = f.readlines()
                self.files = [tmp.strip() for tmp in lines] * epoch_scale
                # here multiply a epoch scale to make the epoch size scalable!!


        if self.split == 'val':
            with open(os.path.join(datalist, 'car_val.txt'),'r') as f:
                lines = f.readlines()
                self.files = [tmp.strip() for tmp in lines]



    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_label_str = self.files[index]


        im_path, boundbox_liststr = img_label_str.strip().split(" ")
        boundbox_and_class_list = [[tmp.split(",")[0],tmp.split(",")[1],tmp.split(",")[2],tmp.split(",")[3],tmp.split(",")[4]] for tmp in boundbox_liststr.split(";")]

        if self.split == "train":
            img = Image.open(os.path.join(self.root, 'leftImg8bit/train', im_path)).convert('RGB')
        elif self.split == "val":
            img = Image.open(os.path.join(self.root, 'leftImg8bit/val', im_path)).convert('RGB')
        else:
            raise ValueError

        if False:
            draw = ImageDraw.Draw(img)
            for bb in boundbox_and_class_list:
                min_x,min_y, max_x, max_y, class_index = bb
                draw.rectangle(((int(min_x), int(min_y)), (int(max_x), int(max_y))), outline="red")
            img.save("vis.jpg", "JPEG")
            return

        img = np.asarray(img)
        height, width, channels = img.shape

        for idx, bb in enumerate(boundbox_and_class_list):
            min_x, min_y, max_x, max_y, class_index = bb

            min_x = float(min_x)/width#normalize 1
            max_x = float(max_x)/width
            min_y = float(min_y)/height
            max_y = float(max_y)/height

            class_index = int(class_index)

            assert not isinstance(min_x, str)
            assert  not isinstance(class_index, str)

            boundbox_and_class_list[idx] = [min_x, min_y, max_x, max_y, class_index]
            #[[xmin, ymin, xmax, ymax, label_ind], ... ], here labl_ind=0, because we only detect car.



        if self.transform is not None:
            target = np.array(boundbox_and_class_list)
            try:
                img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            except:
                import traceback
                traceback.print_exc()
                import ipdb
                ipdb.set_trace()

            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


if __name__ == '__main__':

    from pytorchgo.augmentation.segmentation import PIL2NP
    from torchvision.transforms import Compose, Normalize, ToTensor
    t_loader = CsCarDetection( split='train')

    trainloader = data.DataLoader(t_loader, batch_size=1, num_workers=1, shuffle=True)
    for idx, data in enumerate(trainloader):
        print(idx)
        #image, label = data
        #image = image.numpy()
        #label = label.numpy()
        #from pytorchgo.utils.vis import vis_seg
        #vis_seg(image, label)