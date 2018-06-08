import os
import os.path as osp
import numpy as np
import random
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



if __name__ == '__main__':
    dst = VOCDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
