# Author: Tao Hu <taohu620@gmail.com>
import numpy as np

IMG_MEAN_BGR = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

IMG_MEAN_RGB = np.array((122.67891434,116.66876762,104.00698793), dtype=np.float32)

PASCAL_CLASS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car' , 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']