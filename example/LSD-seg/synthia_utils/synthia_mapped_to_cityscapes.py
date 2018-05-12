# Author: Tao Hu <taohu620@gmail.com>
import os,cv2
from tqdm import tqdm
import numpy as np
from synthia2cityscapes import synthiaid2label
data_dir = "/home/hutao/lab/pytorchgo/example/LSD-seg/data/RAND_CITYSCAPES"
src_dir = os.path.join(data_dir,"GT/LABELS")
target_dir = os.path.join(data_dir,"synthia_mapped_to_cityscapes")

#os.mkdir(target_dir)

def synthia_mapped_to_cityscapes():
    #print synthiaid2label[8].synthia_trainid
    files = os.listdir(src_dir)
    for f in tqdm(files):
        print f
        img = cv2.imread(os.path.join(src_dir,f),cv2.IMREAD_UNCHANGED)
        img = img[:, :, -1]
        print "before transfer: {}".format(str(np.unique(img)))
        img_result = np.copy(img)
        for class_id in synthiaid2label.keys():#notice do not do it in-place!!!
            img_result[img == class_id] = synthiaid2label[class_id].synthia_trainid


        print "after transfer: {}".format(str(np.unique(img_result)))
        cv2.imwrite(os.path.join(target_dir,f),img_result.astype(np.uint8))


synthia_mapped_to_cityscapes()