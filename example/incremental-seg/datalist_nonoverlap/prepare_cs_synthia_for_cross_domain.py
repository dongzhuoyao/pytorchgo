# Author: Tao Hu <taohu620@gmail.com>
import numpy as np
from tqdm import tqdm
import cv2, os, shutil


cs_base = ""
synthia_base = ""

new_label_basedir = "/home/tao/dataset/incremental_seg"


def conduct_filter(filter_func, valtest_filter_func = None, label_name ="class19+1_old"):
    #old data

    if os.path.exists(label_name):
        shutil.rmtree(label_name)
    os.makedirs(label_name)

    if os.path.exists(os.path.join(new_label_basedir, label_name)):
        shutil.rmtree(os.path.join(new_label_basedir, label_name))
    os.makedirs(os.path.join(new_label_basedir, label_name))


    train_num = 0; val_num=0; test_num = 0
    useful_train = 0; useful_val = 0; useful_test = 0
    with open(os.path.join(label_name,"current_incremental_train.txt"),"w") as f:
        for data_type, img_base in zip(["cs","synthia"], [cs_base, synthia_base]):
            lines = open(data_type,"r").readlines()
            lines = [line.strip() for line in lines]

            for line  in tqdm(lines[0:1000], desc="train images for {}/{}".format(label_name,data_type)):
                #if train_num > 100: break

                image, label = line.split()
                img_name = image.split("/")[-1]

                label_image = cv2.imread(os.path.join(img_base, label))
                if filter_func is not None:
                    is_needed, label_image = filter_func(label_image)
                #if not is_needed:
                #    continue
                    if is_needed:
                        useful_train += 1
                cur_label_path = "{}".format(os.path.join(label_name, img_name)).replace("jpg", "png")
                cv2.imwrite(os.path.join(new_label_basedir, cur_label_path), label_image)
                f.write("{} {}\n".format(img_name, cur_label_path))
                train_num += 1



    with open(os.path.join(label_name,"current_incremental_val.txt"),"w") as f:
        for data_type, img_base in zip(["cs","synthia"], [cs_base, synthia_base]):
            lines = open(data_type,"r").readlines()
            lines = [line.strip() for line in lines]

            for line  in tqdm(lines[0:500], desc="val images for {}/{}".format(label_name,data_type)):
                #if train_num > 100: break

                image, label = line.split()
                img_name = image.split("/")[-1]

                label_image = cv2.imread(os.path.join(img_base, label))
                if valtest_filter_func is not None:
                    is_needed, label_image = valtest_filter_func(label_image)
                #if not is_needed:
                #    continue
                    if is_needed:
                        useful_val += 1

                cur_label_path = "{}".format(os.path.join(label_name, img_name)).replace("jpg", "png")
                cv2.imwrite(os.path.join(new_label_basedir, cur_label_path), label_image)
                f.write("{} {}\n".format(img_name, cur_label_path))
                val_num += 1


    with open(os.path.join(label_name, "current_incremental_test.txt"),"w") as f:
        for data_type, img_base in zip(["cs","synthia"], [cs_base, synthia_base]):
            lines = open(data_type,"r").readlines()
            lines = [line.strip() for line in lines]

            for line  in tqdm(lines[0:500], desc="test images for {}/{}".format(label_name,data_type)):
                #if train_num > 100: break

                image, label = line.split()
                img_name = image.split("/")[-1]

                label_image = cv2.imread(os.path.join(img_base, label))
                if valtest_filter_func is not None:
                    is_needed, label_image = valtest_filter_func(label_image)

                #if not is_needed:
                #    continue
                    if is_needed:
                        useful_test += 1

                cur_label_path = "{}".format(os.path.join(label_name, img_name)).replace("jpg", "png")
                cv2.imwrite(os.path.join(new_label_basedir, cur_label_path), label_image)
                f.write("{} {}\n".format(img_name, cur_label_path))
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

def filter10_new(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1,11):#from 1 to 10
        label_img[np.where(label_img_copy==i)] = 0

    ids = set(list(np.unique(label_img)))
    is_needed = True
    if ids == set([255, 0]) or ids == set([0]):
        #print("empty label, skip")
        is_needed = False
        return is_needed, label_img

    for i in range(1,11):
        label_img[np.where(label_img_copy == i+10)] = i
    return is_needed, label_img


#conduct_filter(filter_func=filter10_new, train_slic=[0,5000], valtest_filter_func = None, label_name ="class10+10_new_on_coco")
#conduct_filter(filter_func=filter10_new, train_slic=[0,5000], valtest_filter_func = filter10_new, label_name ="class10+10_singlenetwork_new_on_coco")

#conduct_filter(filter_func=None, train_slic=[0,5000], valtest_filter_func = None, label_name ="class10+10_whole_on_coco")

#conduct_filter(filter_func=filter10_old, train_slic=[0,5000], valtest_filter_func = None, label_name ="cocovoc_10+10_old")

conduct_filter(filter_func=filter10_new, train_slic=[0,5000], valtest_filter_func = None, label_name ="cocovoc_10+10_new")






