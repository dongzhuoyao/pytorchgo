Last Modified: June 26, 2016
German Ros
gros@cvc.uab.es
http://synthia-dataset.net

SYNTHIA: The SYNTHetic collection of Imagery and Annotations

This package contains the SYNTHIA-RAND-CITYSCAPES dataset, with annotations compatible with the CITYSCAPES dataset
https://www.cityscapes-dataset.com/. This set of random images is intended to help to train and test on CITYSCAPES.
The number of classes here presented covers those classes that are used in the test cases of CITYSCAPES, plus some
additions, such as "lanemarking".

Please, if you use this data for research purposes, consider citing our CVPR paper:

German Ros, Laura Sellart, Joanna Materzynska, David Vazquez, Antonio M. Lopez; Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 3234-3243 
@InProceedings{Ros_2016_CVPR,
author = {Ros, German and Sellart, Laura and Materzynska, Joanna and Vazquez, David and Lopez, Antonio M.},
title = {The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2016}
}

DISCLAIMER: 
The data here presented was originally used to train the models of the CVPR paper and to perform an initial evaluation.
If you are looking for the full SYNTHIA dataset, it is *NOT* this one. This version is just an initial subset.


DESCRIPTION:

The package contains the following data,

* RGB: 	    	folder containing standard 1280x760 RGB images used for training
* Depth:    	folder containing 1280x760 unsigned short images. Depth is encoded in any of the 3 channels
	    	in centimetres as an ushort
* GT/COLOR:	folder containing png files (one per image). Annotations are given using a color representation.
		This is mainly provided for visualization and you are not supposed to use them for training.
* GT/LABELS:	folder containing png files (one per image). Annotations are given in two channels. The first
		channel contains the class of that pixel (see the table below). The second channel contains
		the unique ID of the instance for those objects that are dynamic (cars, pedestrians, etc.).


Class		R	G	B	ID
void		0	0	0	0
sky		70	130	180	1
Building	70	70	70	2
Road		128	64	128	3
Sidewalk	244	35	232	4
Fence		64	64	128	5
Vegetation	107	142	35	6
Pole		153	153	153	7
Car		0	0	142	8
Traffic sign	220	220	0	9
Pedestrian	220	20	60	10
Bicycle		119	11	32	11
Motorcycle	0	0	230	12
Parking-slot	250	170	160	13
Road-work	128	64	64	14
Traffic light	250	170	30	15
Terrain		152	251	152	16
Rider		255	0	0	17
Truck		0	0	70	18
Bus		0	60	100	19
Train		0	80	100	20
Wall		102	102	156	21
Lanemarking	102	102	156	22

