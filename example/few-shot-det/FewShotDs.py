# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2

from tensorpack.utils import logger
from tensorpack.dataflow.base import RNGDataFlow
from ss_datalayer import DBInterface


__all__ = ['FewShotDs']


class FewShotDs(RNGDataFlow):
    def __init__(self,name, image_size=(321,321)):
        settings = __import__('ss_settings')
        self.name = name
        profile = getattr(settings, name)
        profile_copy = profile.copy()
        profile_copy['first_shape'] = image_size
        profile_copy['second_shape'] = image_size
        self.dbi = DBInterface(profile)
        self.data_size = len(self.dbi.db_items)
        if "test" in self.name:
            self.data_size = 1000


    def size(self):
        return self.data_size

    @staticmethod
    def class_num():
        return 2

    def get_data(self): # only for one-shot learning
        for i in range(self.data_size):
            first_image_list,first_label_list,second_image, second_label, metadata = self.dbi.next_pair()
            yield [first_image_list,first_label_list,second_image, second_label, metadata]



if __name__ == '__main__':
    from PIL import Image, ImageFont, ImageDraw, ImageEnhance
    ds = FewShotDs("fold0_train")
    ds.reset_state()
    from tensorpack.utils.segmentation.segmentation import apply_mask, visualize_binary_mask
    cur_dir = "fold0_1shot_train_support_masked_images"
    #os.mkdir(cur_dir)
    support_image_size = (320, 320)
    for idx,data in enumerate(ds.get_data()):
        first_images, first_bboxs, second_image, second_bbox, metadata = data
        second_image  = Image.open(second_image).convert('RGB')
        height, width, channels = np.asarray(second_image).shape

        draw = ImageDraw.Draw(second_image)
        for bb in second_bbox:
            min_x, min_y, max_x, max_y = bb
            min_x = float(min_x)* width  # normalize 1
            max_x = float(max_x)*width
            min_y = float(min_y)*height
            max_y = float(max_y)*height

            draw.rectangle(((int(min_x), int(min_y)), (int(max_x), int(max_y))), outline="red")
        second_image.save("second_image.jpg", "JPEG")
        print ("class_name: {}".format(metadata["class_name"]))


