# Author: Tao Hu <taohu620@gmail.com>
import numpy as np



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



class SubtractMeans(object):
    def __init__(self, mean=(104.00699, 116.66877, 122.67892)):#BGR order!
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        image = image.astype(np.float32)
        image -= self.mean
        return image