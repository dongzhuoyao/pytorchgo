# Author: Tao Hu <taohu620@gmail.com>
import os,cv2
import numpy as np
from tqdm import tqdm
id_to_name = {
    0:"background",
    1:"aeroplane",
    2:"bicycle",
    3:"bird",
    4:"boat",
    5:"bottle",
    6:"bus",
    7:"car",
    8:"cat",
    9:"chair",
    10:"cow",
    11:"diningtable",
    12:"dog",
    13:"horse",
    14:"motorbike",
    15:"person",
    16:"plant",
    17:"sheep",
    18:"sofa",
    19:"train",
    20:"tv/monitor"
}

train_aug_path = "train_aug.txt"
val_path = "val.txt"
pascal_root = "/home/hutao/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012"

image_format = os.path.join(pascal_root, 'JPEGImages/{}.jpg')
label_format = os.path.join(pascal_root, "SegmentationClassAug/{}.png")

with open(train_aug_path,"r") as f:
    train_aug_list = [line.strip() for line in f.readlines()]

with open(val_path,"r") as f:
    val_list = [line.strip() for line in f.readlines()]



def filter19(filter_func, result_dir = "class19+1/old", label_dir = "/home/hutao/dataset/incremental_seg/class19+1_old"):
    #old data
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)
    with open(os.path.join(result_dir,"train.txt"),"w") as f:
        for image_id in tqdm(train_aug_list):
            label_image = cv2.imread(label_format.format(image_id),cv2.IMREAD_GRAYSCALE)
            is_needed, label_image = filter_func(label_image)
            if not is_needed:
                continue

            cur_label_path = "{}.png".format(os.path.join(label_dir, image_id))
            cv2.imwrite(cur_label_path,label_image)
            f.write("{}.jpg {}\n".format(image_id, cur_label_path))

    with open(os.path.join(result_dir,"val.txt"),"w") as f:
        for image_id in tqdm(val_list):
            label_image = cv2.imread(label_format.format(image_id), cv2.IMREAD_GRAYSCALE)
            is_needed, label_image = filter_func(label_image)
            if not is_needed:
                continue

            cur_label_path = "{}.png".format(os.path.join(label_dir, image_id))
            cv2.imwrite(cur_label_path, label_image)
            f.write("{}.jpg {}\n".format(image_id, cur_label_path))




def filter19_old(label_img):
    label_img[np.where(label_img == 20)] = 0

    ids = set(list(np.unique(label_img)))
    is_needed = True
    if ids == set([255, 0]):
        print("empty label, skip")
        is_needed = False

    return is_needed, label_img

def filter19_new(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1,20):#from 1 to 19
        label_img[np.where(label_img_copy==i)] = 0

    ids = set(list(np.unique(label_img)))
    is_needed = True
    if ids == set([255, 0]):
        print("empty label, skip")
        is_needed = False
        return is_needed, label_img

    label_img[np.where(label_img_copy == 20)] = 1
    return is_needed, label_img



filter19(filter_func=filter19_new, result_dir = "class19+1/new", label_dir = "/home/hutao/dataset/incremental_seg/class19+1_new")



