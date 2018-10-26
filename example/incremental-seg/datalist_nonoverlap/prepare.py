# Author: Tao Hu <taohu620@gmail.com>
import os,cv2, shutil
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

train_path = "incremental_train.txt"
val_path = "incremental_val.txt"
test_path = "incremental_test.txt"

pascal_root = "/home/tao/dataset/pascalvoc12/VOCdevkit/VOC2012"
new_label_basedir = "/home/tao/dataset/incremental_seg/"

image_format = os.path.join(pascal_root, 'JPEGImages/{}.jpg')
label_format = os.path.join(pascal_root, "SegmentationClassAug/{}.png")

with open(train_path, "r") as f:
    train_list = [line.strip() for line in f.readlines()]

with open(val_path,"r") as f:
    val_list = [line.strip() for line in f.readlines()]

with open(test_path,"r") as f:
    test_list = [line.strip() for line in f.readlines()]



def conduct_filter(filter_func, train_slic=[0,-1], need_remap_val = True, label_name ="class19+1/old"):

    if os.path.exists(label_name):
        shutil.rmtree(label_name)
    os.makedirs(label_name)

    if os.path.exists(os.path.join(new_label_basedir, label_name)):
        shutil.rmtree(os.path.join(new_label_basedir, label_name))
    os.makedirs(os.path.join(new_label_basedir, label_name))



    train_num = 0;val_num=0;test_num=0
    with open(os.path.join(label_name, "current_incremental_train.txt"), "w") as f:
        for image_id in tqdm(train_list[train_slic[0]:train_slic[1]], desc="train images for {}".format(label_name)):
            label_image = cv2.imread(label_format.format(image_id),cv2.IMREAD_GRAYSCALE)
            is_needed, label_image = filter_func(label_image)
            #if not is_needed:
            #    continue

            cur_label_path = "{}.png".format(os.path.join(label_name, image_id))
            cv2.imwrite(os.path.join(new_label_basedir, cur_label_path),label_image)
            f.write("{}.jpg {}\n".format(image_id, cur_label_path))
            train_num += 1

    with open(os.path.join(label_name, "current_incremental_val.txt"), "w") as f:
        for image_id in tqdm(val_list, desc="val images for {}".format(label_name)):
            label_image = cv2.imread(label_format.format(image_id), cv2.IMREAD_GRAYSCALE)

            if need_remap_val:
                is_needed, label_image = filter_func(label_image)
            #if not is_needed:
            #    continue

            cur_label_path = "{}.png".format(os.path.join(label_name, image_id))
            cv2.imwrite(os.path.join(new_label_basedir, cur_label_path), label_image)
            f.write("{}.jpg {}\n".format(image_id, cur_label_path))
            val_num += 1

    with open(os.path.join(label_name, "current_incremental_test.txt"), "w") as f:
        for image_id in tqdm(test_list, desc="test images for {}".format(label_name)):
            label_image = cv2.imread(label_format.format(image_id), cv2.IMREAD_GRAYSCALE)

            if need_remap_val:
                is_needed, label_image = filter_func(label_image)
            #if not is_needed:
            #    continue

            cur_label_path = "{}.png".format(os.path.join(label_name, image_id))
            cv2.imwrite(os.path.join(new_label_basedir, cur_label_path), label_image)
            f.write("{}.jpg {}\n".format(image_id, cur_label_path))
            test_num += 1

    print "train num={}".format(train_num)
    print "val num={}".format(val_num)
    print "test num={}".format(test_num)




def filter19_old(label_img):
    label_img[np.where(label_img == 20)] = 0

    ids = set(list(np.unique(label_img)))
    is_needed = True
    if ids == set([255, 0]) or ids == set([0]):
        print("empty label, skip")
        is_needed = False

    return is_needed, label_img

def filter19_new(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1,20):#from 1 to 19
        label_img[np.where(label_img_copy==i)] = 0

    ids = set(list(np.unique(label_img)))
    is_needed = True
    if ids == set([255, 0]) or ids == set([0]):
        print("empty label, skip")
        is_needed = False
        return is_needed, label_img

    label_img[np.where(label_img_copy == 20)] = 1
    return is_needed, label_img


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


def filter15_old(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(16, 21):#16~20
        label_img[np.where(label_img_copy==i)] = 0

    ids = set(list(np.unique(label_img)))
    is_needed = True
    if ids == set([255, 0]) or ids == set([0]):
        print("empty label, skip")
        is_needed = False

    return is_needed, label_img

def filter15_new(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1,16):#from 1 to 15
        label_img[np.where(label_img_copy==i)] = 0

    ids = set(list(np.unique(label_img)))
    is_needed = True
    if ids == set([255, 0]) or ids == set([0]):
        print("empty label, skip")
        is_needed = False
        return is_needed, label_img

    for i in range(1,6):
        label_img[np.where(label_img_copy == i+15)] = i
    return is_needed, label_img


def filter15_gradual_new16(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1,16):#from 1 to 15
        label_img[np.where(label_img_copy==i)] = 0

    label_img[np.where(label_img_copy == 16)] = 1
    label_img[np.where(label_img_copy == 17)] = 0
    label_img[np.where(label_img_copy == 18)] = 0
    label_img[np.where(label_img_copy == 19)] = 0
    label_img[np.where(label_img_copy == 20)] = 0
    return True, label_img

def filter15_gradual_new17(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1,16):#from 1 to 15
        label_img[np.where(label_img_copy==i)] = 0

    label_img[np.where(label_img_copy == 16)] = 0
    label_img[np.where(label_img_copy == 17)] = 1
    label_img[np.where(label_img_copy == 18)] = 0
    label_img[np.where(label_img_copy == 19)] = 0
    label_img[np.where(label_img_copy == 20)] = 0
    return True, label_img

def filter15_gradual_new18(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1,16):#from 1 to 15
        label_img[np.where(label_img_copy==i)] = 0

    label_img[np.where(label_img_copy == 16)] = 0
    label_img[np.where(label_img_copy == 17)] = 0
    label_img[np.where(label_img_copy == 18)] = 1
    label_img[np.where(label_img_copy == 19)] = 0
    label_img[np.where(label_img_copy == 20)] = 0
    return True, label_img

def filter15_gradual_new19(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1,16):#from 1 to 15
        label_img[np.where(label_img_copy==i)] = 0

    label_img[np.where(label_img_copy == 16)] = 0
    label_img[np.where(label_img_copy == 17)] = 0
    label_img[np.where(label_img_copy == 18)] = 0
    label_img[np.where(label_img_copy == 19)] = 1
    label_img[np.where(label_img_copy == 20)] = 0
    return True, label_img

def filter15_gradual_new20(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1,16):#from 1 to 15
        label_img[np.where(label_img_copy==i)] = 0

    label_img[np.where(label_img_copy == 16)] = 0
    label_img[np.where(label_img_copy == 17)] = 0
    label_img[np.where(label_img_copy == 18)] = 0
    label_img[np.where(label_img_copy == 19)] = 0
    label_img[np.where(label_img_copy == 20)] = 1
    return True, label_img


def filter10_gradual_new15(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1,11):#from 1 to 10
        label_img[np.where(label_img_copy==i)] = 0



    label_img[np.where(label_img_copy == 11)] = 1
    label_img[np.where(label_img_copy == 12)] = 2
    label_img[np.where(label_img_copy == 13)] = 3
    label_img[np.where(label_img_copy == 14)] = 4
    label_img[np.where(label_img_copy == 15)] = 5

    label_img[np.where(label_img_copy == 16)] = 0
    label_img[np.where(label_img_copy == 17)] = 0
    label_img[np.where(label_img_copy == 18)] = 0
    label_img[np.where(label_img_copy == 19)] = 0
    label_img[np.where(label_img_copy == 20)] = 0
    return True, label_img


def filter10_gradual_new20(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1, 11):  # from 1 to 10
        label_img[np.where(label_img_copy == i)] = 0

    label_img[np.where(label_img_copy == 11)] = 0
    label_img[np.where(label_img_copy == 12)] = 0
    label_img[np.where(label_img_copy == 13)] = 0
    label_img[np.where(label_img_copy == 14)] = 0
    label_img[np.where(label_img_copy == 15)] = 0

    label_img[np.where(label_img_copy == 16)] = 1
    label_img[np.where(label_img_copy == 17)] = 2
    label_img[np.where(label_img_copy == 18)] = 3
    label_img[np.where(label_img_copy == 19)] = 4
    label_img[np.where(label_img_copy == 20)] = 5
    return True, label_img


def single16(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1, 16):  # from 1 to 10
        label_img[np.where(label_img_copy == i)] = 0

    label_img[np.where(label_img_copy == 16)] = 1

    for i in range(17, 21):  # from 1 to 10
        label_img[np.where(label_img_copy == i)] = 0

    return True, label_img



def single17(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1, 17):  # from 1 to 10
        label_img[np.where(label_img_copy == i)] = 0

    label_img[np.where(label_img_copy == 17)] = 1

    for i in range(18, 21):  # from 1 to 10
        label_img[np.where(label_img_copy == i)] = 0

    return True, label_img

def single18(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1, 18):  # from 1 to 10
        label_img[np.where(label_img_copy == i)] = 0

    label_img[np.where(label_img_copy == 18)] = 1

    for i in range(19, 21):  # from 1 to 10
        label_img[np.where(label_img_copy == i)] = 0

    return True, label_img

def single19(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1, 19):  # from 1 to 10
        label_img[np.where(label_img_copy == i)] = 0

    label_img[np.where(label_img_copy == 19)] = 1

    for i in range(20, 21):  # from 1 to 10
        label_img[np.where(label_img_copy == i)] = 0

    return True, label_img

def single20(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1, 20):  # from 1 to 10
        label_img[np.where(label_img_copy == i)] = 0

    label_img[np.where(label_img_copy == 20)] = 1


    return True, label_img


def total16(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(17, 21):  # from 1 to 10
        label_img[np.where(label_img_copy == i)] = 0

    return True, label_img

def total17(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(18, 21):  # from 1 to 10
        label_img[np.where(label_img_copy == i)] = 0

    return True, label_img

def total18(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(19, 21):  # from 1 to 10
        label_img[np.where(label_img_copy == i)] = 0

    return True, label_img

def total19(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(20, 21):  # from 1 to 10
        label_img[np.where(label_img_copy == i)] = 0

    return True, label_img





#conduct_filter(filter_func=filter10_old, train_slic=[0,5000], need_remap_val=True, label_name="class10+10_old")
#conduct_filter(filter_func=filter10_new, train_slic=[5000,-1], need_remap_val=False, label_name="class10+10_new")



conduct_filter(filter_func=filter10_old, train_slic=[0,5000], need_remap_val=True, label_name="class10+10_singlenetwork_old")
conduct_filter(filter_func=filter10_new, train_slic=[5000,-1], need_remap_val=True, label_name="class10+10_singlenetwork_new")




















"""
conduct_filter(filter_func=filter15_old, result_dir ="class15+5/old", label_dir ="/home/hutao/dataset/incremental_seg/class15+5_old")
conduct_filter(filter_func=filter15_new, result_dir ="class15+5/new", label_dir ="/home/hutao/dataset/incremental_seg/class15+5_new")


conduct_filter(filter_func=filter19_old, result_dir ="class19+1/old", label_dir ="/home/hutao/dataset/incremental_seg/class19+1_old")
conduct_filter(filter_func=filter19_new, result_dir ="class19+1/new", label_dir ="/home/hutao/dataset/incremental_seg/class19+1_new")



conduct_filter(filter_func=filter15_gradual_new16, result_dir ="class15_gradual/new16", label_dir ="/home/hutao/dataset/incremental_seg/class15_gradual_new16")
conduct_filter(filter_func=filter15_gradual_new17, result_dir ="class15_gradual/new17", label_dir ="/home/hutao/dataset/incremental_seg/class15_gradual_new17")
conduct_filter(filter_func=filter15_gradual_new18, result_dir ="class15_gradual/new18", label_dir ="/home/hutao/dataset/incremental_seg/class15_gradual_new18")
conduct_filter(filter_func=filter15_gradual_new19, result_dir ="class15_gradual/new19", label_dir ="/home/hutao/dataset/incremental_seg/class15_gradual_new19")
conduct_filter(filter_func=filter15_gradual_new20, result_dir ="class15_gradual/new20", label_dir ="/home/hutao/dataset/incremental_seg/class15_gradual_new20")



conduct_filter(filter_func=filter10_gradual_new15, result_dir ="class10_gradual/new15", label_dir ="/home/hutao/dataset/incremental_seg/class15_gradual_new15")
conduct_filter(filter_func=filter10_gradual_new20, result_dir ="class10_gradual/new20", label_dir ="/home/hutao/dataset/incremental_seg/class15_gradual_new20")


conduct_filter(filter_func=single16, result_dir ="single16", label_dir ="/home/hutao/dataset/incremental_seg/single16")
conduct_filter(filter_func=single17, result_dir ="single17", label_dir ="/home/hutao/dataset/incremental_seg/single17")
conduct_filter(filter_func=single18, result_dir ="single18", label_dir ="/home/hutao/dataset/incremental_seg/single18")
conduct_filter(filter_func=single19, result_dir ="single19", label_dir ="/home/hutao/dataset/incremental_seg/single19")
conduct_filter(filter_func=single20, result_dir ="single20", label_dir ="/home/hutao/dataset/incremental_seg/single20")

conduct_filter(filter_func=total16, result_dir ="total16", label_dir ="/home/hutao/dataset/incremental_seg/total16")
conduct_filter(filter_func=total17, result_dir ="total17", label_dir ="/home/hutao/dataset/incremental_seg/total17")
conduct_filter(filter_func=total18, result_dir ="total18", label_dir ="/home/hutao/dataset/incremental_seg/total18")
conduct_filter(filter_func=total19, result_dir ="total19", label_dir ="/home/hutao/dataset/incremental_seg/total19")


"""






