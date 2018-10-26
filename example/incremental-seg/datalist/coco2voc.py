# Author: Tao Hu <taohu620@gmail.com>

# Author: Tao Hu <taohu620@gmail.com>

from pycocotools.coco import COCO
from pycocotools import mask
from tensorpack.utils.segmentation.segmentation import  visualize_label
import numpy as np
import os
import os,cv2
import shutil
from tqdm import tqdm

coco_dataset = "/data2/dataset/coco"
detection_json_train = "/data2/dataset/annotations/instances_train2014.json"
detection_json_val = "/data2/dataset/annotations/instances_val2014.json"
train_dir = "/data2/dataset/coco/train2014"
val_dir = "/data2/dataset/coco/val2014"

result_dir = "/home/hutao/dataset/incremental_seg/coco2voc"
image_dir = os.path.join(result_dir,"image")
label_dir = os.path.join(result_dir,"label")

if os.path.isdir(image_dir):
    shutil.rmtree(image_dir)
if os.path.isdir(label_dir):
    shutil.rmtree(label_dir)


os.makedirs(image_dir)
os.makedirs(label_dir)



voc_ids_list = [5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
#sofa=couch


voc_ids_set = set(voc_ids_list)
coco_to_voc_dict = {}
for voc_id, coco_id in enumerate(voc_ids_list):
    coco_to_voc_dict[coco_id] = voc_id+1# start from 1



def proceed(detection_json, coco_img_dir, txt_name="coco2voc_train.txt"):
    f = open(txt_name, "w")
    _coco = COCO(detection_json)
    result = _coco.getCatIds()
    catToImgs = _coco.catToImgs
    imgToAnns = _coco.imgToAnns
    img_ids_set = set()
    for id,coco_id in enumerate(voc_ids_list):
        img_ids_set = img_ids_set | set(catToImgs[coco_id])
    img_ids_list = list(img_ids_set)

    for img_id in tqdm(img_ids_list[1:]):
        img = _coco.loadImgs(img_id)[0]
        origin_img = cv2.imread(os.path.join(coco_img_dir, img['file_name']))
        annIds = _coco.getAnnIds(imgIds=img_id)
        img_mask = np.zeros((img['height'], img['width'],1),dtype=np.uint8)

        for annId in annIds:
            ann = _coco.loadAnns(annId)[0]
            if ann['category_id'] in voc_ids_set:
                # polygon
                if type(ann['segmentation']) == list:
                    for _instance in ann['segmentation']:
                            rle = mask.frPyObjects([_instance], img['height'], img['width'])
                            m = mask.decode(rle)
                            img_mask[np.where(m == 1)] = coco_to_voc_dict[ann['category_id']]
                # mask
                else:# mostly is aeroplane
                    if type(ann['segmentation']['counts']) == list:
                        rle = mask.frPyObjects([ann['segmentation']], img['height'], img['width'])
                    else:
                        rle = [ann['segmentation']]
                    m = mask.decode(rle)
                    img_mask[np.where(m == 1)] = coco_to_voc_dict[ann['category_id']]

        f.write("{} {}\n".format(os.path.join("image",img['file_name']),os.path.join(label_dir,img['file_name'].replace("jpg","png"))))

        cv2.imwrite(os.path.join(image_dir,img['file_name']),origin_img)
        cv2.imwrite(os.path.join(label_dir, img['file_name'].replace("jpg","png")), img_mask[:,:,0])#single channel


        if False:
            cv2.imwrite("cv.jpg", origin_img)
            cv2.imwrite("mask.jpg",img_mask)
            cv2.imshow("im", origin_img)
            cv2.imshow("color-label", visualize_label(img_mask[:,:,0]))
            cv2.waitKey(0)
    f.close()

print("proceed train")
proceed(detection_json_train, train_dir, "coco2voc_train.txt")

print("proceed val")
proceed(detection_json_val, val_dir, "coco2voc_val.txt")








