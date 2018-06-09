import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data


class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, img_transform = None, label_transform = None, augmentation = None,
                 max_iters=None,  mirror=True, ):
        self.root = root
        self.list_path = list_path
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
	        self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.img_transform = img_transform
        self.augmentation = augmentation
        self.label_transform = label_transform

        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        image = np.asarray(image, np.float32)

        if self.img_transform is not None:
                image = self.img_transform(image)

        if self.augmentation is not None:
            image, label = self.augmentation((image, label))

        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name



class CSDataSet(data.Dataset):
    def __init__(self, root='/home/hutao/dataset/cityscapes', list_path="datalist/cityscapes", name="train", img_transform = None, label_transform = None, augmentation = None,
                 max_iters=None,  mirror=True, ):
        self.root = root
        self.list_path = list_path
        self.is_mirror = mirror
        self.name = name
        if "train" in self.name:
            self.data_pairs = [(data_pair[0].strip(), data_pair[1].strip()) for data_pair in zip(open(os.path.join(list_path, "cityscapes_imagelist_train.txt")), open(os.path.join(list_path, "cityscapes_labellist_train.txt")))]
        if not max_iters==None:
	        self.data_pairs = self.data_pairs * int(np.ceil(float(max_iters) / len(self.data_pairs)))

        self.img_transform = img_transform
        self.augmentation = augmentation
        self.label_transform = label_transform

        self.files = []
        for img_path,label_path in self.data_pairs:
            if "train" in self.name:
                img_file = osp.join(self.root, "leftImg8bit/train", img_path)
                label_file = osp.join(self.root, "gtFine/train", label_path)
            elif "val" in self.name:
                img_file = osp.join(self.root, "leftImg8bit/val", img_path)
                label_file = osp.join(self.root, "gtFine/val", label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": img_path
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        image = np.asarray(image, np.float32)

        if False:
            from tensorpack.utils.segmentation.segmentation import visualize_label
            cv2.imwrite("img.jpg", np.concatenate((image, visualize_label(label)), axis=1))
            print(np.unique(label))

        if self.img_transform is not None:
                image = self.img_transform(image)

        if self.augmentation is not None:
            image, label = self.augmentation((image, label))

        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name




if __name__ == '__main__':
    dst = CSDataSet()
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
       pass

