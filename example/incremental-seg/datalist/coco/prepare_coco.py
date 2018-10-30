# Author: Tao Hu <taohu620@gmail.com>
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask
from collections import Counter
from tqdm import tqdm
import cv2, os, shutil

"""
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
#trainid2catstr = {value:key for key,value in catstr2trainid.items()}
#catid2catstr = {catid: trainid2catstr[catid2trainid[catid]] for catid in  catid2trainid.keys()}
#trainid2catid = {value:key for key,value in catid2trainid.items()}
"""



catstr2trainid = {
'toilet':62,
'teddy bear':78,
'cup':42,
'bicycle':2,
'kite':34,
'carrot':52,
'stop sign':12,
'tennis racket':39,
'donut':55,
'snowboard':32,
'sandwich':49,
'motorcycle':4,
'oven':70,
'keyboard':67,
'scissors':77,
'airplane':5,
'couch':58,
'mouse':65,
'fire hydrant':11,
'boat':9,
'apple':48,
'sheep':19,
'horse':18,
'banana':47,
'baseball glove':36,
'tv':63,
'traffic light':10,
'chair':57,
'bowl':46,
'microwave':69,
'bench':14,
'book':74,
'elephant':21,
'orange':50,
'tie':28,
'clock':75,
'bird':15,
'knife':44,
'pizza':54,
'fork':43,
'hair drier':79,
'frisbee':30,
'umbrella':26,
'bottle':40,
'bus':6,
'bear':22,
'vase':76,
'toothbrush':80,
'spoon':45,
'train':7,
'sink':72,
'potted plant':59,
'handbag':27,
'cell phone':68,
'toaster':71,
'broccoli':51,
'refrigerator':73,
'laptop':64,
'remote':66,
'surfboard':38,
'cow':20,
'dining table':61,
'hot dog':53,
'car':3,
'sports ball':33,
'skateboard':37,
'dog':17,
'bed':60,
'cat':16,
'person':1,
'skis':31,
'giraffe':24,
'truck':8,
'parking meter':13,
'suitcase':29,
'cake':56,
'wine glass':41,
'baseball bat':35,
'backpack':25,
'zebra':23,
}


def generate_image_mask(_coco, img_mask, annId, cat_dict=catid2trainid):
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
        ann = _coco.loadAnns(annId)[0]

        img_mask, ann = generate_image_mask(_coco, img_mask, annId)

    return img_file_name, img_mask


from pycocotools.coco import COCO
_coco = COCO("/data2/dataset/annotations/instances_train2014.json")
_coco_val = COCO("/data2/dataset/annotations/instances_val2014.json")
COCO_PATH = '/data2/dataset/coco'
image_format = os.path.join(COCO_PATH, 'train2014/{}')
label_format = os.path.join(COCO_PATH, "val2014/{}")

coco_result_path = "/home/hutao/dataset/incremental_coco"



def conduct_filter(filter_func, result_dir ="class40+40/old", label_dir ="/home/hutao/dataset/incremental_coco/class40+40_old"):
    #old data

    if os.path.exists(label_dir):
        shutil.rmtree(label_dir)
    os.makedirs(label_dir)

    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)



    train_num = 0;val_num=0
    with open(os.path.join(result_dir,"train.txt"),"w") as f:
        for key,value in tqdm(_coco.imgs.items(), desc="train images for {}".format(result_dir)):
            #if train_num > 100: break
            image_id = value['id']
            img_name = value['file_name']
            _, label_image = generate_mask(_coco, image_id)
            is_needed, label_image = filter_func(label_image)
            #if not is_needed:
            #    continue

            cur_label_path = "{}".format(os.path.join(label_dir, img_name)).replace("jpg", "png")
            cv2.imwrite(cur_label_path,label_image)
            f.write("{} {}\n".format(img_name, cur_label_path))
            train_num += 1

    with open(os.path.join(result_dir,"val.txt"),"w") as f:
        for key,value in tqdm(_coco_val.imgs.items(), desc="val images for {}".format(result_dir)):
            #if val_num > 100: break
            image_id = value['id']
            img_name = value['file_name']
            _, label_image = generate_mask(_coco_val, image_id)
            is_needed, label_image = filter_func(label_image)
            #if not is_needed:
            #    continue

            cur_label_path = "{}".format(os.path.join(label_dir, img_name)).replace("jpg", "png")
            cv2.imwrite(cur_label_path,label_image)
            f.write("{} {}\n".format(img_name, cur_label_path))
            val_num += 1

    print "train num={}".format(train_num)
    print "val num={}".format(val_num)





def filter40_old(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(41, 81):#16~20
        label_img[np.where(label_img_copy==i)] = 0

    ids = set(list(np.unique(label_img)))
    is_needed = True
    if ids == set([255, 0]) or ids == set([0]):
        print("empty label, skip")
        is_needed = False

    return is_needed, label_img

def filter40_new(label_img):
    label_img_copy = np.copy(label_img)
    for i in range(1,41):#from 1 to 40
        label_img[np.where(label_img_copy==i)] = 0

    ids = set(list(np.unique(label_img)))
    is_needed = True
    if ids == set([255, 0]) or ids == set([0]):
        print("empty label, skip")
        is_needed = False
        return is_needed, label_img

    for i in range(1,41):
        label_img[np.where(label_img_copy == i+40)] = i
    return is_needed, label_img

def filter80(label_img):
    return True,label_img


#conduct_filter(filter_func=filter80, result_dir ="class80", label_dir ="/home/hutao/dataset/incremental_coco/class80")


#conduct_filter(filter_func=filter40_old, result_dir ="class40+40/old", label_dir ="/home/hutao/dataset/incremental_coco/class40+40_old")
#conduct_filter(filter_func=filter40_new, result_dir ="class40+40/new", label_dir ="/home/hutao/dataset/incremental_coco/class40+40_new")





