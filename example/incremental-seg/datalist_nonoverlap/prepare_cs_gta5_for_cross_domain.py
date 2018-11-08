# Author: Tao Hu <taohu620@gmail.com>
import numpy as np
from tqdm import tqdm
import cv2, os, shutil
from PIL import Image

cs_base = "/home/tao/dataset/cityscapes"
gta5_base = "/home/tao/dataset/GTA5"

new_label_basedir = "/home/tao/dataset/incremental_seg"

id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                      19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                      26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}


def get_lines(data_type, split_type="train"):
    if data_type == "cs":
        if split_type == "train":
            lines = open("cityscapes_list/cs_train.txt", "r").readlines()
        elif split_type == "val":
            lines = open("cityscapes_list/cs_val.txt", "r").readlines()
        else:
            raise ValueError("errr")
    elif  data_type == "gta5":
        if split_type == "train":
            lines = open("gta5_list/gta5_train.txt", "r").readlines()
        elif split_type == "val":
            lines = open("gta5_list/gta5_val.txt", "r").readlines()
        else:
            raise ValueError("error")

    lines = [line.strip() for line in lines]
    return lines

def read_image(img_base, label, data_type, split_type):
    if data_type == "cs" :
        if split_type == "train":
            label_image = cv2.imread(os.path.join(img_base, "gtFine/train", label))
        elif split_type == "val":
            label_image = cv2.imread(os.path.join(img_base, "gtFine/val", label))
        else:
            raise ValueError("error")
        return label_image

    elif data_type == "gta5":
        label_image =  Image.open(os.path.join(img_base, label))
        label_image = np.asarray(label_image, np.float32)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label_image.shape, dtype=np.float32)
        for k, v in id_to_trainid.items():
            label_copy[label_image == k] = v
        return label_copy
    else:
        raise ValueError("error")




def conduct_filter(filter_func, valtest_filter_func = None, label_name ="class19+1_old"):
    #old data

    if os.path.exists(label_name):
        shutil.rmtree(label_name)
    os.makedirs(label_name)

    if os.path.exists(os.path.join(new_label_basedir, "{}_{}".format(label_name, "cs"))):
        shutil.rmtree(os.path.join(new_label_basedir, "{}_{}".format(label_name, "cs")))
    os.makedirs(os.path.join(new_label_basedir, "{}_{}".format(label_name, "cs")))

    if os.path.exists(os.path.join(new_label_basedir, "{}_{}".format(label_name, "gta5"))):
        shutil.rmtree(os.path.join(new_label_basedir, "{}_{}".format(label_name, "gta5")))
    os.makedirs(os.path.join(new_label_basedir, "{}_{}".format(label_name, "gta5")))


    train_num = 0; val_num=0; test_num = 0
    useful_train = 0; useful_val = 0; useful_test = 0
    with open(os.path.join(label_name,"current_incremental_train.txt"),"w") as f:
        for data_type, img_base in zip(["cs","gta5"], [cs_base, gta5_base]):
            lines = get_lines(data_type=data_type, split_type="train")

            for line  in tqdm(lines[0:1000], desc="train images for {}/{}".format(label_name,data_type)):
                #if train_num > 100: break

                image, label = line.split()
                img_name = image.split("/")[-1]

                label_image = read_image(img_base=img_base,label=label,data_type=data_type, split_type="train")
                if filter_func is not None:
                    is_needed, label_image = filter_func(label_image)
                #if not is_needed:
                #    continue
                    if is_needed:
                        useful_train += 1


                cur_label_path = os.path.join("{}_{}".format(label_name, data_type), img_name).replace("jpg", "png")
                cv2.imwrite(os.path.join(new_label_basedir, cur_label_path), label_image)
                if data_type == "cs":
                    f.write("{} {}\n".format(os.path.join("gtFine/train",image), cur_label_path))
                elif data_type =="gta5":
                    f.write("{} {}\n".format(image, cur_label_path))
                else:
                    raise ValueError("error")
                train_num += 1



    with open(os.path.join(label_name,"current_incremental_val.txt"),"w") as f:
        for data_type, img_base in zip(["cs","gta5"], [cs_base, gta5_base]):
            lines = get_lines(data_type=data_type, split_type="val")

            for line  in tqdm(lines[0:500], desc="val images for {}/{}".format(label_name,data_type)):
                #if train_num > 100: break

                image, label = line.split()
                img_name = image.split("/")[-1]

                label_image = read_image(img_base=img_base, label=label, data_type=data_type, split_type="val")
                if valtest_filter_func is not None:
                    is_needed, label_image = valtest_filter_func(label_image)
                #if not is_needed:
                #    continue
                    if is_needed:
                        useful_val += 1

                cur_label_path = os.path.join("{}_{}".format(label_name, data_type), img_name).replace("jpg", "png")
                cv2.imwrite(os.path.join(new_label_basedir, cur_label_path), label_image)
                if data_type == "cs":
                    f.write("{} {}\n".format(os.path.join("gtFine/val",image), cur_label_path))
                elif data_type =="gta5":
                    f.write("{} {}\n".format(image, cur_label_path))
                else:
                    raise ValueError("error")
                val_num += 1


    with open(os.path.join(label_name, "current_incremental_test.txt"),"w") as f:
        for data_type, img_base in zip(["cs","gta5"], [cs_base, gta5_base]):

            lines = get_lines(data_type=data_type, split_type="val")
            for line  in tqdm(lines[0:500], desc="test images for {}/{}".format(label_name,data_type)):
                #if train_num > 100: break

                image, label = line.split()
                img_name = image.split("/")[-1]

                label_image = read_image(img_base=img_base, label=label, data_type=data_type,  split_type="val")
                if valtest_filter_func is not None:
                    is_needed, label_image = valtest_filter_func(label_image)

                #if not is_needed:
                #    continue
                    if is_needed:
                        useful_test += 1

                cur_label_path = os.path.join("{}_{}".format(label_name, data_type), img_name).replace("jpg", "png")
                cv2.imwrite(os.path.join(new_label_basedir, cur_label_path), label_image)
                if data_type == "cs":
                    f.write("{} {}\n".format(os.path.join("gtFine/val",image), cur_label_path))
                elif data_type =="gta5":
                    f.write("{} {}\n".format(image, cur_label_path))
                else:
                    raise ValueError("error")
                test_num += 1

    print "train num={}, useful num={}".format(train_num, useful_train)
    print "val num={}, useful num={}".format(val_num, useful_val)
    print "test num={}, useful num={}".format(test_num, useful_test)


def filter10_old(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(11, 21):#11~20
        label_img[np.where(label_img_copy==i)] = 0

    ids = set(list(np.unique(label_img)))
    is_needed = True
    if ids == set([255, 0]) or ids == set([0]):
        #print("empty label, skip")
        is_needed = False
    return is_needed, label_img

def filter8_new(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1,11):#from 1 to 10
        label_img[np.where(label_img_copy==i)] = 0

    ids = set(list(np.unique(label_img)))
    is_needed = True
    if ids == set([255, 0]) or ids == set([0]):
        #print("empty label, skip")
        is_needed = False
        return is_needed, label_img

    for i in range(1,9):
        label_img[np.where(label_img_copy == i+10)] = i
    return is_needed, label_img

#conduct_filter(filter_func=filter10_old, valtest_filter_func = None, label_name ="cs_gta5_10+10_old")
#conduct_filter(filter_func=filter8_new,  valtest_filter_func = filter8_new, label_name ="cs_gta5_10+8_new")
#conduct_filter(filter_func=None,  valtest_filter_func = None, label_name ="cs_gta5_10+8_whole")
conduct_filter(filter_func=filter8_new,  valtest_filter_func = filter8_new, label_name ="cs_gta5_10+8_single_new")


