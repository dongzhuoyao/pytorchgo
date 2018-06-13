# Author: Tao Hu <taohu620@gmail.com>
import numpy as np
from PIL import Image
import collections
import torch
import cv2,random
import numbers

from PIL import Image, ImageOps

class RandomScale():
    def __init__(self):
        pass
    def __call__(self, data):
        image, label = data
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return (image, label)




class PascalPadding():
    def __init__(self,size):
        self.crop_h = size[0]
        self.crop_w = size[1]
        self.ignore_label = 255

    def __call__(self, data):
        image, label = data
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        return (image, label)


class PIL2NP_BOTH(object):
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask  = data
        return (np.asarray(image), np.asarray(mask))




class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, data):
        img, mask = data


        assert img.shape[:2] == mask.shape

        h, w = mask.shape
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        assert w >= tw and h >= th

        x1 = random.randint(0, h - th)
        y1 = random.randint(0, w - tw)
        return (img[x1:x1+th,y1:y1+tw], mask[x1:x1+th,y1:y1+tw])


############################## single image ###########################################

class PIL2NP(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return np.asarray(image)




class RGB2BGR(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image = image.astype(np.float32)
        image = image[:, :, ::-1]  # RGB -> BGR
        return image

class Value255to0(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image = image.astype(np.uint8)
        image[image==255] = 0
        return image

class ToLabel(object):
    def __call__(self, inputs):
        # tensors = []
        # for i in inputs:
        # tensors.append(torch.from_numpy(np.array(i)).long())
        tensors = torch.from_numpy(np.array(inputs)).long()
        return tensors

class SubtractMeans(object):
    def __init__(self, mean=(104.00699, 116.66877, 122.67892)):#BGR order!
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        image = image.astype(np.float32)
        image -= self.mean
        return image

class PIL_Scale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)