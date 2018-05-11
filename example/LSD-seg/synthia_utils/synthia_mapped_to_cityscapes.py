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
    files = os.listdir(src_dir)
    for f in tqdm(files):
        print f
        img = cv2.imread(os.path.join(src_dir,f),cv2.IMREAD_UNCHANGED)
        img = img[:, :, -1]
        print np.unique(img)
        for class_id in synthiaid2label.keys():
            img[img==class_id] = synthiaid2label[class_id].synthia_trainid
        print np.unique(img)
        cv2.imwrite(os.path.join(target_dir,f),img.astype(np.uint8))


synthia_mapped_to_cityscapes()