import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image


class SynthiaDataSet(data.Dataset):
    def __init__(self, root, list_path, label_list_path ,max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror

        self.img_ids = [i_id.strip() for i_id in open(list_path)]


        self.files = []

        self.label_list_path = label_list_path
        self.label_img_ids = [i_id.strip() for i_id in open(label_list_path)]

        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_img_ids = self.label_img_ids * int(np.ceil(float(max_iters) / len(self.label_img_ids)))




        for image,label in zip(self.img_ids,self.label_img_ids):
            img_file = osp.join(self.root,  image)
            label_file = osp.join(self.root,  label)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": image
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        if False:
            cv2.imshow("img", image)
            cv2.imshow("label", label)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dst = SynthiaDataSet(root="../data/SYNTHIA",
                         list_path="synthia_list/SYNTHIA_imagelist_train.txt",
                         label_list_path="synthia_list/SYNTHIA_labellist_train.txt")
    trainloader = data.DataLoader(dst, batch_size=4)

    for i, data in enumerate(trainloader):
        imgs, labels, size, name = data

        import cv2
        from tensorpack.utils.segmentation.segmentation import predict_slider, visualize_label, predict_scaler
        img = torchvision.utils.make_grid(imgs).numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, ::-1]+128

        label = torchvision.utils.make_grid(labels.unsqueeze(1)).numpy()
        label = np.transpose(label, (1, 2, 0))
        #plt.imshow(img)
        #plt.show()
        cv2.imshow("source image", img.astype(np.uint8))
        cv2.imshow("source label", visualize_label(label[:,:,0]))
        cv2.waitKey(10000)
