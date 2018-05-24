# README #

**my running result is 28% mIoU**


==> Loading FCN8s_LSD model file: logs/MODEL-LSD_CFG-Adam_LR_0.00001000/model_best.pth.tar
==> Evaluating with CityScapes validation
tensorpack mIoU: 0.234553658922
tensorpack mean_accuracy: 0.337092400824
tensorpack accuracy: 0.663078122518
('Num classes', 19)
===>road:	45.95
===>sidewalk:	19.18
===>building:	61.01
===>wall:	4.82
===>fence:	0.46
===>pole:	20.07
===>light:	2.02
===>sign:	9.52
===>vegetation:	75.04
===>terrain:	0.0
===>sky:	76.94
===>person:	41.75
===>rider:	14.29
===>car:	48.81
===>truck:	0.0
===>bus:	6.03
===>train:	0.0
===>motocycle:	7.17
===>bicycle:	12.6
===> mIoU: 23.46


==> Loading FCN8s_LSD model file: logs/MODEL-LSD_CFG-Adam_LR_0.00001000/model_best.pth.tar
==> Evaluating with CityScapes validation
tensorpack mIoU: 0.286795885361
tensorpack mean_accuracy: 0.385299141096
tensorpack accuracy: 0.798419604166
('Num classes', 19)
===>road:       78.47
===>sidewalk:   29.99
===>building:   75.9
===>wall:       6.9
===>fence:      0.85
===>pole:       21.73
===>light:      10.29
===>sign:       14.43
===>vegetation: 76.42
===>terrain:    0.0
===>sky:        77.73
===>person:     42.49
===>rider:      14.84
===>car:        62.78
===>truck:      0.0
===>bus:        11.48
===>train:      0.0
===>motocycle:  7.05
===>bicycle:    13.58
===> mIoU: 28.68

class16-mIoU= 34.06




Code for Semantic Segmentation for Unsupervised Domain Adaptation

Prerequisites:

	Install pytorch (Version 0.2) and torchvision
	pip install  http://download.pytorch.org/whl/cu80/torch-0.2.0.post2-cp27-cp27mu-manylinux1_x86_64.whl
	Install fcn (pip install fcn)
	Install OpenCV (pip install opencv-python)

Datasets:

We will need two datasets for our experiments - SYNTHIA and CITYSCAPES. Please download the datasets into data folder from the following links

Please download SYNTHIA-RAND-CITYSCAPES subset of the SYNTHIA dataset.
	SYNTHIA: http://synthia-dataset.net/download-2/

	CITYSCAPES: https://www.cityscapes-dataset.com/

cd  into ./data, and run the following:

    ln -s /your-path/RAND_CITYSCAPES/ RAND_CITYSCAPES
    ln -s /your-path/cityscapes/ cityscapes

To run the code, go to code folder and run the following command:

	python run_script.py

This assumes that the data is downloaded and paths are set accordingly. Options can be modified directly in the train.py script.

Please change the dataroot path, and logdir path accordingly. This will run the code and save the models in logdir folder. 

To evaluate the trained model, run 

	python eval_cityscapes.py --dataroot [] --model\_file [] -- method []

## TODO

GTAV dataset preparation

    https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/segmentation
    
Cityscapes image should rescaled!!!