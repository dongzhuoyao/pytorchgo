# Author: Tao Hu <taohu620@gmail.com>

#void, sky, building, road, sidewalk, \
#fence, vegetation, pole, car,traffic sign, \
#pedestrian, bicycle, motorcycle, parking-slot,road-work,
#traffic light, terrain, rider, truck, bus
#train, wall, lanemarking

from collections import namedtuple

city2common = namedtuple( 'city2common' , [
    'name',
    'cityscapes_id',
    'common_id',
    ] )

synthia2common = namedtuple( 'synthia2common' , [
    'name',
    'synthia_id',
    'common_id',
    ] )

city2common_table = [
    city2common('unlabeled', 0, 255),
    city2common('ego vehicle',            1, 255),
    city2common('rectification border',       2, 255),
    city2common('out of roi',           3, 255),
    city2common('static',       4, 255),
    city2common('dynamic',          5, 255),
    city2common('ground',     6, 255),
    city2common('road',           7, 0),
    city2common('sidewalk',            8, 1),
    city2common('parking',   9, 255),
    city2common('rail track',     10, 255),
    city2common('building',        11, 2),
    city2common('wall',     12, 3),
    city2common('fence',   13, 4),
    city2common('guard rail',      14, 255),
    city2common('bridge',  15, 255),
    city2common('tunnel',        16, 255),
    city2common('pole',          17, 5),
    city2common('polegroup',          18, 255),
    city2common('traffic light',            19, 6),
    city2common('traffic sign',          20, 7),
    city2common('vegetation',           21, 8),
    city2common('terrain',    22, 255),
    city2common('sky',    23, 9),
    city2common('person',    24, 10),
    city2common('rider',    25, 11),
    city2common('car',    26, 12),
    city2common('truck',    27, 255),
    city2common('bus',    28, 13),
    city2common('caravan',    29, 255),
    city2common('trailer',    30, 255),
    city2common('train',    31, 255),
    city2common('motorcycle',    32, 14),
    city2common('bicycle',    33, 15),
    city2common('license plate',    -1, 255),
]



synthia2common_table = [
    synthia2common('void',           0, 255),
    synthia2common('sky',            1, 9),
    synthia2common('building',       2, 2),
    synthia2common('road',           3, 0),
    synthia2common('sidewalk',       4, 1),
    synthia2common('fence',          5, 4),
    synthia2common('vegetation',     6, 8),
    synthia2common('pole',           7, 5),
    synthia2common('car',            8, 12),
    synthia2common('traffic sign',   9, 7),
    synthia2common('pedestrian',     10, 10),
    synthia2common('bicycle',        11, 15),
    synthia2common('motorcycle',     12, 14),
    synthia2common('parking slot',   13, 255),
    synthia2common('road-work',      14, 255),
    synthia2common('traffic light',  15, 6),
    synthia2common('terrain',        16, 255),
    synthia2common('rider',          17, 11),
    synthia2common('truck',          18, 255),
    synthia2common('bus',            19, 13),
    synthia2common('train',          20, 255),
    synthia2common('wall',           21, 3),
    synthia2common('lanemarking',    22, 255),
]


# cityscapes's class 9,14,16 are missed in SYNTHIA!!!!


synthia2common_dict = { label.synthia_id : label for label in synthia2common_table }
city2common_dict = { label.cityscapes_id : label for label in city2common_table }



