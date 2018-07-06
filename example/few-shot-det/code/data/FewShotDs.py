# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

from data import FewShotVOC
from PIL import Image
__all__ = ['FewShotDs','FewShotVOCDataset']

import torch,random
import torch.utils.data as data
from pytorchgo.utils.constant import IMG_MEAN
from pytorchgo.utils.rect import FloatBox,IntBox
from pytorchgo.utils.viz import draw_boxes

from pytorchgo.utils.constant import PASCAL_CLASS


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

    def __init__(self, name, image_size=(300, 300), second_image_augs=None,channel6=False,channel4=False, channel5=False):
        self.name = name
        self.image_size = image_size
        profile = getattr(FewShotVOC, name)
        self.params = profile
        self.dbi = FewShotVOC.DBInterface(profile)
        self.data_size = len(self.dbi.db_items)
        self.init_randget(profile['read_mode'])

        self.second_image_augs = second_image_augs
        self.channel6 = channel6
        self.channel5 = channel5
        self.channel4 = channel4

    def init_randget(self, read_mode):
        self.rand_gen = random.Random()
        if read_mode == 'shuffle':
            self.rand_gen.seed()
        elif read_mode == 'deterministic':
            self.rand_gen.seed(1385)  # >>>Do not change<<< Fixed seed for deterministic mode.



    def __getitem__(self, index):

        def get_item(idx):
            imgset, second_index = self.dbi.db_items[idx]  # query image index
            set_indices = list(range(second_index)) + list(
                range(second_index + 1, len(imgset.image_items)))  # exclude second_index
            assert (len(set_indices) >= self.params['k_shot'])
            self.rand_gen.shuffle(set_indices)#TODO
            first_index = set_indices[:self.params['k_shot']]  # support set image indexes(may be multi-shot~)
            class_id = imgset.image_items[second_index].obj_ids[0]
            metadata = {
                'class_id': class_id,
                'class_name': PASCAL_CLASS[class_id - 1],
                'second_image_path': imgset.image_items[second_index].img_path,
            }


            return [imgset.image_items[v].img_path for v in first_index], \
                   [imgset.image_items[v].bbox for v in first_index], \
                   imgset.image_items[second_index].img_path, \
                   imgset.image_items[second_index].bbox, metadata


        first_images, first_bboxs, second_image, second_bbox, metadata = get_item(index)
        second_image = Image.open(second_image).convert('RGB')
        second_image = np.asarray(second_image, np.float32)

        k_shot = len(first_images)
        output_first_images = []
        metadata_origin_first_images = []
        output_first_masks = []
        output_first_masked_images = []
        output_first_masked_images_concat = []


        def rgb_shit(image):
            image = image[:, :, ::-1]  # change to BGR
            image -= IMG_MEAN
            return image.transpose((2, 0, 1))  # W,H,C->C,W,H

        for k in range(k_shot):
            first_image = first_images[k]
            first_image = Image.open(first_image).convert('RGB')
            first_image = np.asarray(first_image,dtype=np.float32)

            height, width, channels = first_image.shape
            first_mask = np.zeros((height,width,1),np.float32)
            bboxs = first_bboxs[k]
            origin_first_image = np.copy(first_image)
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

                intbox = IntBox(min_x, min_y, max_x, max_y)
                intbox.clip_by_shape((height, width))
                origin_first_image = draw_boxes(origin_first_image,[intbox],color=(255,0,0))

            first_mask = cv2.resize(first_mask,self.image_size)#resize, this resize don't keep extra 3rd dim, so you must extend dim yourself again!
            first_mask = first_mask[:,:,np.newaxis]
            first_image = cv2.resize(first_image, self.image_size)#resize
            metadata_origin_first_images.append(origin_first_image.astype(np.uint8))

            output_first_images.append(first_image)
            output_first_masks.append(first_mask)
            if self.channel4:
                masked = first_mask
            elif self.channel5:
                masked = first_mask
            else:
                masked = first_image * first_mask
            output_first_masked_images.append(masked)

            if self.channel6 or self.channel4:
                first_image = rgb_shit(first_image)
                masked = rgb_shit(masked)
                ttt = np.concatenate((first_image,masked),axis=0)
            elif self.channel5:
                tmp = np.copy(masked)
                masked_exchanged = np.zeros(masked.shape, masked.dtype)
                masked_exchanged[np.where(tmp==0)] = 1
                masked_exchanged[np.where(tmp == 1)] = 0

                ttt = np.concatenate((first_image, masked, masked_exchanged), axis=2)
            else:
                masked = rgb_shit(masked)
                ttt = masked

            output_first_masked_images_concat.append(ttt)

        second_image = cv2.resize(second_image, self.image_size)  # resize
        second_origin_image = np.copy(second_image).astype(np.uint8)

        for i, bb in enumerate(second_bbox):
            bb.append(0)#add default class, notice here!!!

        if self.second_image_augs is not None:
            second_bbox_np = np.stack(second_bbox, axis=0)
            img, boxes, labels = self.second_image_augs(second_image, second_bbox_np[:, :4], second_bbox_np[:, 4])
            #img = img[:, :, ::-1]  # change to BGR
            if False:
                height, width, channels = img.shape
                images_cv = np.copy(img)
                for _ in range(boxes.shape[0]):
                    min_x = boxes[_,0]
                    min_y = boxes[_,1]
                    max_x = boxes[_,2]
                    max_y = boxes[_,3]
                    min_x = float(min_x) * width  # normalize 1
                    max_x = float(max_x) * width
                    min_y = float(min_y) * height
                    max_y = float(max_y) * height

                    floatBox = FloatBox(min_x, min_y, max_x, max_y)
                    #floatBox.clip_by_shape((image_size, image_size))
                    images_cv = draw_boxes(images_cv, [floatBox], color=(255, 0, 0))

                cv2.imwrite("second_image.jpg", images_cv)
                print("class_name: {}".format(metadata["class_name"]))

            img -= IMG_MEAN
            img = img[:, :, (2, 1, 0)]#back to rgb
            second_image =  img.transpose((2, 0, 1))
            second_bbox = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        else:
            second_image = rgb_shit(second_image)





        output_first_masked_images_concat = np.stack(output_first_masked_images_concat, axis=0)

        output_first_masked_images_concat = np.squeeze(output_first_masked_images_concat)#only for one-shot!!!

        metadata['second_origin_image'] = second_origin_image
        metadata['metadata_origin_first_images'] = metadata_origin_first_images

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
    from utils.augmentations import SSDAugmentation
    dataset = FewShotVOCDataset(name="fold0_1shot_train",second_image_augs=SSDAugmentation(512))
    data_loader = data.DataLoader(dataset, batch_size=1, num_workers=1,
                                  shuffle=True, pin_memory=True,collate_fn=detection_collate)

    for idx,data in enumerate(data_loader):
        print("ok")


