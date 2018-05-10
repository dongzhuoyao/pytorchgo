# Author: Tao Hu <taohu620@gmail.com>

#void, sky, building, road, sidewalk, \
#fence, vegetation, pole, car,traffic sign, \
#pedestrian, bicycle, motorcycle, parking-slot,road-work,
#traffic light, terrain, rider,truck, bus
#train, wall, lanemarking

from collections import namedtuple

Label = namedtuple( 'Label' , [
    'synthia_name',
    'synthia_id',
    'synthia_trainid',
    'cityscapes_trainid'
    ] )

labels = [
    Label('void',           0, 255 ,255),
    Label('sky',            1, 10 ,10),
    Label('building',       2, 2, 2),
    Label('road',           3, 0, 0),
    Label('sidewalk',       4, 1, 1),
    Label('fence',          5, 4, 4),
    Label('vegetation',     6, 8, 8),
    Label('pole',           7, 5, 5),
    Label('car',            8, 13, 13),
    Label('traffic sign',   9, 7, 7),
    Label('pedestrian',     10, 11, 11),
    Label('bicycle',        11, 18, 18),
    Label('motorcycle',     12, 17, 17),
    Label('parking slot',   13, 255, 255),
    Label('road-work',      14, 255, 255),
    Label('traffic light',  15, 6, 6),
    Label('terrain',        16, 255, 255),
    Label('rider',          17, 12, 12),
    Label('truck',          18, 255, 255),
    Label('bus',            19, 15, 15),
    Label('train',          20, 255, 255),
    Label('wall',           21, 3, 3),
    Label('lanemarking',    22, 255, 255),
]

# cityscapes's class 9,14,16 are missed in SYNTHIA!!!!


synthiaid2label = { label.synthia_id : label for label in labels }




