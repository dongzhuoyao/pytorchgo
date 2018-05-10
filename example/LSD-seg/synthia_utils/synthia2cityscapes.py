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
    Label('sky',            1, 0 ,10),
    Label('building',       2, 1, 2),
    Label('road',           3, 2, 0),
    Label('sidewalk',       4, 3, 1),
    Label('fence',          5, 4, 4),
    Label('vegetation',     6, 5, 8),
    Label('pole',           7, 6, 5),
    Label('car',            8, 7, 13),
    Label('traffic sign',   9, 8, 7),
    Label('pedestrian',     10, 9, 11),
    Label('bicycle',        11, 10, 18),
    Label('motorcycle',     12, 11, 17),
    Label('parking slot',   13, 255, 255),
    Label('road-work',      14, 255, 255),
    Label('traffic light',  15, 12, 6),
    Label('terrain',        16, 255, 255),
    Label('rider',          17, 13, 12),
    Label('truck',          18, 255, 255),
    Label('bus',            19, 14, 15),
    Label('train',          20, 255, 255),
    Label('wall',           21, 15, 3),
    Label('lanemarking',    22, 255, 255),
]


synthiaid2label = { label.synthia_id : label for label in labels }




