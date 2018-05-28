# Author: Tao Hu <taohu620@gmail.com>
import numpy as np
from PIL import Image
import collections
import torch

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