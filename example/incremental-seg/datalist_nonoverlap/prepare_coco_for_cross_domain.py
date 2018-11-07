# Author: Tao Hu <taohu620@gmail.com>
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask
from collections import Counter
from tqdm import tqdm
import cv2, os, shutil

catid2trainid = {
1:1,
2:2,
3:3,
4:4,
5:5,
6:6,
7:7,
8:8,
9:9,
10:10,
11:11,
13:12,
14:13,
15:14,
16:15,
17:16,
18:17,
19:18,
20:19,
21:20,
22:21,
23:22,
24:23,
25:24,
27:25,
28:26,
31:27,
32:28,
33:29,
34:30,
35:31,
36:32,
37:33,
38:34,
39:35,
40:36,
41:37,
42:38,
43:39,
44:40,
46:41,
47:42,
48:43,
49:44,
50:45,
51:46,
52:47,
53:48,
54:49,
55:50,
56:51,
57:52,
58:53,
59:54,
60:55,
61:56,
62:57,
63:58,
64:59,
65:60,
67:61,
70:62,
72:63,
73:64,
74:65,
75:66,
76:67,
77:68,
78:69,
79:70,
80:71,
81:72,
82:73,
84:74,
85:75,
86:76,
87:77,
88:78,
89:79,
90:80,
}

voc_ids_list = [5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72] #catid of coco indexed by pascal voc

coco_catid2voc_trainid_dict = {}
def init_coco_catid2voc_trainid_dict():
    for i in range(91):
        coco_catid2voc_trainid_dict[i] = 0# set as background first.
    for voc_trainid, coco_catid in enumerate(voc_ids_list):
        coco_catid2voc_trainid_dict[coco_catid] = voc_trainid + 1 #start from 1 for non-background classes.

init_coco_catid2voc_trainid_dict()




def generate_image_mask(_coco, img_mask, annId, cat_dict=coco_catid2voc_trainid_dict):
    height,width,_ =img_mask.shape
    ann = _coco.loadAnns(annId)[0]

    # polygon
    if type(ann['segmentation']) == list:
        for _instance in ann['segmentation']:
            rle = mask.frPyObjects([_instance], height, width)
            m = mask.decode(rle)
            img_mask[np.where(m == 1)] = cat_dict[ann['category_id']]
    # mask
    else:  # mostly is aeroplane
        if type(ann['segmentation']['counts']) == list:
            rle = mask.frPyObjects([ann['segmentation']], height, width)
        else:
            rle = [ann['segmentation']]
        m = mask.decode(rle)
        img_mask[np.where(m == 1)] = cat_dict[ann['category_id']]



    return img_mask, ann



def generate_mask(_coco, img_id):
    img = _coco.loadImgs(img_id)[0]
    img_file_name = img['file_name']
    annIds = _coco.getAnnIds(imgIds=img_id)
    img_mask = np.zeros((img['height'], img['width'], 1), dtype=np.uint8)

    for annId in annIds:
        img_mask, ann = generate_image_mask(_coco, img_mask, annId)

    return img_file_name, img_mask


from pycocotools.coco import COCO
_coco = COCO("/home/tao/dataset/coco14/annotations/instances_train2014.json")
_coco_val = COCO("/home/tao/dataset/coco14/annotations/instances_val2014.json")
COCO_PATH = '/home/tao/dataset/coco14/'
image_format = os.path.join(COCO_PATH, 'train2014/{}')
label_format = os.path.join(COCO_PATH, "val2014/{}")

new_label_basedir = "/home/tao/dataset/incremental_seg"


def conduct_filter(filter_func, train_slic=[0,-1], valtest_filter_func = None, label_name ="class19+1_old"):
    #old data

    if os.path.exists(label_name):
        shutil.rmtree(label_name)
    os.makedirs(label_name)

    if os.path.exists(os.path.join(new_label_basedir, label_name)):
        shutil.rmtree(os.path.join(new_label_basedir, label_name))
    os.makedirs(os.path.join(new_label_basedir, label_name))


    train_num = 0;val_num=0;test_num = 0
    useful_train = 0; useful_val = 0; useful_test = 0
    with open(os.path.join(label_name,"coco_incremental_train.txt"),"w") as f:
        for key,value in tqdm(_coco.imgs.items()[train_slic[0]:train_slic[1]], desc="train images for {}".format(label_name)):
            #if train_num > 100: break
            image_id = value['id']
            img_name = value['file_name']
            _, label_image = generate_mask(_coco, image_id)
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

    with open(os.path.join(label_name,"coco_incremental_val.txt"),"w") as f:
        for key,value in tqdm(_coco_val.imgs.items()[:582], desc="val images for {}".format(label_name)):
            #if val_num > 100: break
            image_id = value['id']
            img_name = value['file_name']
            _, label_image = generate_mask(_coco_val, image_id)
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


    with open(os.path.join(label_name, "coco_incremental_test.txt"),"w") as f:
        for key,value in tqdm(_coco_val.imgs.items()[582:582+1449], desc="test images for {}".format(label_name)):
            #if val_num > 100: break
            image_id = value['id']
            img_name = value['file_name']
            _, label_image = generate_mask(_coco_val, image_id)
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




#conduct_filter(filter_func=None, train_slic=[0,5000], valtest_filter_func = None, label_name ="cocovoc_whole") #common test, validatation data
#conduct_filter(filter_func=filter10_old, train_slic=[0,5000], valtest_filter_func = filter10_old, label_name ="cocovoc_10+10_old")
#conduct_filter(filter_func=filter10_new, train_slic=[0,5000], valtest_filter_func = None, label_name ="cocovoc_10+10_new")
conduct_filter(filter_func=filter10_new, train_slic=[0,5000], valtest_filter_func = filter10_new, label_name ="cocovoc_10+10_single_new")




