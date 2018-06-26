# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2

from tensorpack.utils import logger
from tensorpack.dataflow.base import RNGDataFlow
import ss_datalayer


__all__ = ['FewShotDs']


class FewShotDs(RNGDataFlow):
    def __init__(self,name, image_size=(321,321)):
        settings = __import__('ss_settings')
        self.name = name
        profile = getattr(settings, name)
        profile_copy = profile.copy()
        profile_copy['first_shape'] = image_size
        profile_copy['second_shape'] = image_size
        dbi = ss_datalayer.DBInterface(profile)
        self.data_size = len(dbi.db_items)
        if "test" in self.name:
            self.data_size = 1000
        self.PLP = ss_datalayer.PairLoaderProcess(dbi, profile_copy)

    def size(self):
        return self.data_size

    @staticmethod
    def class_num():
        return 2

    def get_data(self): # only for one-shot learning
        for i in range(self.data_size):
            first_image_list,first_label_list,second_image, second_label, metadata = self.PLP.load_next_frame()
            yield [first_image_list,first_label_list,second_image, second_label, metadata]



if __name__ == '__main__':
    ds = FewShotDs("fold0_5shot_test")
    from tensorpack.utils.segmentation.segmentation import apply_mask, visualize_binary_mask
    cur_dir = "fold0_5shot_test_support_masked_images"
    #os.mkdir(cur_dir)
    support_image_size = (320, 320)
    for idx,data in enumerate(ds.get_data()):
        first_images = data[0]
        first_masks = data[1]
        second_images = data[2]
        second_masks = data[3]
        metadata = data[4]
        class_id = metadata['class_id']
        for kk in range(5):
            first_image = cv2.imread(first_images[kk], cv2.IMREAD_COLOR)
            first_label = cv2.imread(first_masks[kk], cv2.IMREAD_GRAYSCALE)
            first_label = np.equal(first_label, class_id).astype(np.uint8)
            first_image = cv2.resize(first_image, support_image_size)
            first_label = cv2.resize(first_label, support_image_size, interpolation=cv2.INTER_NEAREST)
            first_image_masked = visualize_binary_mask(first_image, first_label, color=(255, 0, 0),class_num=2)
            cv2.imwrite(os.path.join(cur_dir,"{}_shot{}.jpg".format(idx,kk)), first_image_masked)

        print "{} {}   {}".format(idx, ','.join(metadata['image1_name']),metadata['image2_name'])
