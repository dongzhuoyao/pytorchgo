import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data

new_label_basedir = "/home/tao/dataset/incremental_seg/"

class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, img_transform = None, label_transform = None, augmentation = None,
                 max_iters=None,  mirror=True, img_prefix = 'JPEGImages' ):
        self.root = root
        self.list_path = list_path
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if not max_iters==None:
	        self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.img_transform = img_transform
        self.augmentation = augmentation
        self.label_transform = label_transform

        self.files = []
        for ttt in self.img_ids:
            try:
                image_id, val_img_path = ttt
            except:
                raise
            #inference img_file path from image_name
            if "COCO" in image_id:
                if "train" in image_id:
                    foldername = "train2014"
                elif "val" in image_id:
                    foldername = "val2014"
                elif "test" in image_id:
                    foldername = "test2014"
                else:
                    raise
                img_file = osp.join("/home/tao/dataset/coco14", foldername, "%s" % image_id)
            else:
                img_file = osp.join(self.root, img_prefix, "%s" % image_id)
            label_file = osp.join(new_label_basedir, val_img_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": image_id
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        datafiles["img"] = datafiles["img"].replace("png","jpg")
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        image = np.asarray(image, np.float32)

        origin_image = np.copy(image)
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
        if "train" in self.list_path:
            return image.copy(), label.copy(), np.array(size), name
        else:
            return image.copy(), image.copy(), label.copy(), np.array(size), name


class CoCoDataSet(data.Dataset):
    def __init__(self, list_path, img_transform = None, label_transform = None, augmentation = None,
                 max_iters=None, mirror=True, ):
        self.img_root = "/data2/dataset/coco"
        self.list_path = list_path
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if not max_iters==None:
	        self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.img_transform = img_transform
        self.augmentation = augmentation
        self.label_transform = label_transform

        self.files = []
        for image_name, val_img_path in self.img_ids:
            if "train" in self.list_path:
                img_file = osp.join(self.img_root, "train2014/{}".format(image_name))
            elif "val" in self.list_path:
                img_file = osp.join(self.img_root, "val2014/{}".format(image_name))
            label_file = val_img_path
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": image_name
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
            cv2.imwrite("img.jpg", np.concatenate((image, visualize_label(label,class_num=41)), axis=1))
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
    #dst = VOCDataSet("/home/hutao/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012", list_path="datalist/class19+1/new/val_1449.txt")
    #dst = CoCoDataSet(list_path="datalist/coco/class40+40/new/val.txt")
    dst = VOCDataSet("/home/hutao/dataset/incremental_seg/coco2voc", list_path="datalist/coco2voc_val.txt",img_prefix="")
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels,_,_,_ = data
        if i == -1:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]

