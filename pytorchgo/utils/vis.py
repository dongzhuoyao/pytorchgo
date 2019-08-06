# Author: Tao Hu <taohu620@gmail.com>
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def save_as_image_grid(images,figure_size_inches=(14,3), rows=1, titles=None, save_img_name="grid.png"):
    #https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, np.ceil(n_images / float(rows)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(figure_size_inches[0],figure_size_inches[1])#(np.array(fig.get_size_inches()) * n_images)
    plt.tight_layout()
    plt.savefig(save_img_name)


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
