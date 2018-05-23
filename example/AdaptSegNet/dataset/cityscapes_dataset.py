import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image

DATA_DIRECTORY_TARGET = './data/cityscapes'

train_img = './dataset/cityscapes_list/cityscapes_imagelist_train.txt'
train_label = './dataset/cityscapes_list/cityscapes_labellist_train.txt'

val_img = './dataset/cityscapes_list/cityscapes_imagelist_val.txt'
val_label = './dataset/cityscapes_list/cityscapes_labellist_val.txt'


class cityscapesDataSet(data.Dataset):
    def __init__(self,  max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = DATA_DIRECTORY_TARGET
        if set == "train":
            self.list_path = train_img
            self.label_list_path = train_label
        elif set == "val":
            self.list_path = val_img
            self.label_list_path = val_label
        else:
            raise

        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        self.label_img_ids = [i_id.strip() for i_id in open(self.label_list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_img_ids = self.label_img_ids * int(np.ceil(float(max_iters) / len(self.label_img_ids)))

        self.files = []
        self.set = set
        # for split in ["train", "trainval", "val"]:
        for image,label in zip(self.img_ids,self.label_img_ids):
            img_path = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, image))
            label_path = osp.join(self.root, "gtFine/%s/%s" % (self.set, label))
            self.files.append({
                "img": img_path,
                "label":label_path,
                "name": img_path
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        label = Image.open(datafiles["label"])

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
