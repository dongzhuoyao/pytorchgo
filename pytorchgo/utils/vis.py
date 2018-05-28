# Author: Tao Hu <taohu620@gmail.com>
import torchvision
import numpy as np

def vis_seg(imgs, labels, waitkey = 10000):
    import cv2
    from tensorpack.utils.segmentation.segmentation import predict_slider, visualize_label, predict_scaler
    img = torchvision.utils.make_grid(imgs).numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img[:, :, ::-1] + 128

    label = torchvision.utils.make_grid(labels.unsqueeze(1)).numpy()
    label = np.transpose(label, (1, 2, 0))
    # plt.imshow(img)
    # plt.show()
    cv2.imshow("source image", img.astype(np.uint8))
    cv2.imshow("source label", visualize_label(label[:, :, 0]))
    cv2.waitKey(waitkey)