# Pytorchgo Model Zoo

## Details you should notice

Except recover the weight, you also need recover the image loading pattern.


## caffe model

expect different preprocessing than the other models in the PyTorch model zoo. 
Images should be in BGR format in the range [0, 255], 
and the following BGR values should then be subtracted from each pixel: [103.939, 116.779, 123.68].

|Arch|prototxt|caffe model|pytorch model|pytorch code|
|----|----|----|----|----|
|vgg16 init|||[vgg16_from_caffe.pth](https://dongzhuoyao.oss-cn-qingdao.aliyuncs.com/vgg16_from_caffe.pth)||
|Deeplabv1-largeFOV pretrained|[train.prototxt](http://www.cs.jhu.edu/~alanlab/ccvl/DeepLab-LargeFOV/train.prototxt) |[train2_iter_8000.caffemodel](http://www.cs.jhu.edu/~alanlab/ccvl/DeepLab-LargeFOV/train2_iter_8000.caffemodel)|[pth](https://dongzhuoyao.oss-cn-qingdao.aliyuncs.com/deeplabv1_init_model.pth)|?|
|Deeplabv2-VGG|||||
|Deeplabv2-ResNet101 init|||[MS_DeepLab_resnet_pretrained_COCO_init.pth](https://dongzhuoyao.oss-cn-qingdao.aliyuncs.com/MS_DeepLab_resnet_pretrained_COCO_init.pth)|?|

## Related Links

* [https://github.com/marvis/pytorch-caffe](https://github.com/marvis/pytorch-caffe)
* [https://github.com/ruotianluo/pytorch-resnet](https://github.com/ruotianluo/pytorch-resnet)
