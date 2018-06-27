# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2


from . import FewShotVOC
from PIL import Image
__all__ = ['FewShotDs','FewShotVOCDataset']

import torch
import torch.utils.data as data
from pytorchgo.utils.constant import IMG_MEAN

class FewShotVOCDataset(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, name, image_size=(300, 300)):
        self.name = name
        self.image_size = image_size
        profile = getattr(FewShotVOC, name)
        self.dbi = FewShotVOC.DBInterface(profile)
        self.data_size = len(self.dbi.db_items)
        if "test" in self.name:
            self.data_size = 1000

    def __getitem__(self, index):
        first_images, first_bboxs, second_image, second_bbox, metadata = self.dbi.next_pair()
        second_image = Image.open(second_image).convert('RGB')
        second_image = np.asarray(second_image,np.float32)
        if False:
            height, width, channels = second_image.shape

            draw = ImageDraw.Draw(second_image)
            for bb in second_bbox:
                min_x = bb[0]
                min_y = bb[1]
                max_x = bb[2]
                max_y = bb[3]
                min_x = float(min_x) * width  # normalize 1
                max_x = float(max_x) * width
                min_y = float(min_y) * height
                max_y = float(max_y) * height

                draw.rectangle(((int(min_x), int(min_y)), (int(max_x), int(max_y))), outline="red")
            second_image.save("second_image.jpg", "JPEG")
            print ("class_name: {}".format(metadata["class_name"]))

        k_shot = len(first_images)
        output_first_images = []
        output_first_masks = []
        output_first_masked_images = []
        output_first_masked_images_concat = []
        for k in range(k_shot):
            first_image = first_images[k]
            first_image = Image.open(first_image).convert('RGB')
            first_image = np.asarray(first_image)

            height, width, channels = first_image.shape
            first_mask = np.zeros((height,width,1),np.float32)
            bboxs = first_bboxs[k]
            for bbox in bboxs:
                min_x = bbox[0]
                min_y = bbox[1]
                max_x = bbox[2]
                max_y = bbox[3]
                min_x = int(min_x * width)  # normalize 1
                max_x = int(max_x * width)
                min_y = int(min_y * height)
                max_y = int(max_y * height)
                first_mask[min_y:max_y,min_x:max_x] = 1

            first_mask = cv2.resize(first_mask,self.image_size)#resize, this resize don't keep extra 3rd dim, so you must extend dim yourself again!
            first_mask = first_mask[:,:,np.newaxis]
            first_image = cv2.resize(first_image, self.image_size)#resize

            output_first_images.append(first_image)
            output_first_masks.append(first_mask)
            masked = first_image * first_mask
            output_first_masked_images.append(masked)
            ttt = np.concatenate((first_image,masked),axis=2)
            ttt = np.transpose(ttt,(2,0,1))#W,H,C->C,W,H
            output_first_masked_images_concat.append(ttt)

        second_image = cv2.resize(second_image, self.image_size)  # resize

        second_image = second_image[:, :, ::-1]  # change to BGR
        second_image -= IMG_MEAN
        second_image = second_image.transpose((2, 0, 1))#W,H,C->C,W,H


        for i, bb in enumerate(second_bbox):
            bb.append(0)#add default class, notice here!!!


        output_first_masked_images_concat = np.stack(output_first_masked_images_concat, axis=0)

        return torch.from_numpy(output_first_masked_images_concat.copy()), torch.from_numpy(second_image.copy()), second_bbox, metadata

    def __len__(self):
        return self.data_size


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    first_images = []
    second_bboxes = []
    second_images = []
    metadata_list = []
    for sample in batch:
        first_images.append(sample[0])
        second_images.append(sample[1])
        second_bboxes.append(torch.FloatTensor(sample[2]))
        metadata_list.append(sample[3])
    return torch.stack(first_images, 0), torch.stack(second_images, 0), second_bboxes, metadata_list

if __name__ == '__main__':
    from PIL import Image, ImageFont, ImageDraw, ImageEnhance
    dataset = FewShotVOCDataset(name="fold0_train")
    data_loader = data.DataLoader(dataset, batch_size=4, num_workers=1,
                                  shuffle=True, pin_memory=True,collate_fn=detection_collate)

    from tensorpack.utils.segmentation.segmentation import apply_mask, visualize_binary_mask
    cur_dir = "fold0_1shot_train_support_masked_images"
    #os.mkdir(cur_dir)
    for idx,data in enumerate(data_loader):
        output_first_images, output_first_masks, output_first_masked_images, second_image, second_bbox, metadata= data[0]
        cv2.imwrite("first_masked_image.jpg",output_first_images[0])
        print ("class_name: {}".format(metadata["class_name"]))
        print("ok")


