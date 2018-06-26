import numpy as np
import os.path as osp
from pytorchgo.utils.map_util import Map


# Classes in pascal dataset
PASCAL_CATS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car' , 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa',
               'train', 'tv/monitor']

PASCAL_PATH= '/data2/dataset/VOCdevkit'
image_size = (321, 321)


def get_cats(split, fold, num_folds=4):
    '''
      Returns a list of categories (for training/test) for a given fold number

      Inputs:
        split: specify train/val
        fold : fold number, out of num_folds
        num_folds: Split the set of image classes to how many folds. In BMVC paper, we use 4 folds

    '''
    num_cats = len(PASCAL_CATS)
    assert(num_cats%num_folds==0)
    val_size = int(num_cats/num_folds)
    assert(fold<num_folds)
    val_set = [ fold*val_size+v for v in range(val_size)]
    train_set = [x for x in range(num_cats) if x not in val_set]
    if split=='train':
        return [PASCAL_CATS[x] for x in train_set] 
    else:
        return [PASCAL_CATS[x] for x in val_set] 


default_profile = Map(
                ###############################################
                k_shot=1,
                first_shape=None,
                second_shape=None,
                output_type=None,
                read_mode=None, # Either "Shuffle" (for training) or "Deterministic" (for testing, random seed fixed)
                default_pascal_cats = PASCAL_CATS,
                default_coco_cats = None,
                pascal_cats = PASCAL_CATS,
                pascal_path = PASCAL_PATH,
                coco_cats = None,
                worker_num = 4)


foldall_train = Map(default_profile,
                    read_mode='shuffle',
                    image_sets='pascal_train',
                    pascal_cats = PASCAL_CATS,
                    first_shape=image_size,
                    second_shape=image_size) # original code is second_shape=None),TODO

foldall_1shot_test = Map(default_profile,
                         db_cycle = 1000,
                         read_mode='deterministic',
                         image_sets='pascal_val',
                         pascal_cats = PASCAL_CATS,
                         first_shape=image_size,
                         second_shape=image_size,
                         k_shot=1) # original code is second_shape=None),TODO

foldall_5shot_test = Map(default_profile,
                         db_cycle = 1000,
                         read_mode='deterministic',
                         image_sets='pascal_val',
                         pascal_cats = PASCAL_CATS,
                         first_shape=image_size,
                         second_shape=image_size,
                         k_shot=5) # original code is second_shape=None),TODO


#### fold 0 ####

# Setting for training (on **training images**)
fold0_train = Map(default_profile,
                  read_mode='shuffle',
                  image_sets='pascal_train',
                  pascal_cats = get_cats('train',0),
                  first_shape=image_size,
                  second_shape=image_size) # original code is second_shape=None),TODO

fold0_5shot_train = Map(fold0_train,k_shot=5)

# Setting for testing on **test images** in unseen image classes (in total 5 classes), 5-shot
fold0_5shot_test = Map(default_profile,
                       db_cycle = 1000,
                       read_mode='deterministic',
                       image_sets='pascal_val',
                       pascal_cats = get_cats('val',0),
                       first_shape=image_size,
                       second_shape=image_size,
                       k_shot=5)

#### fold 1 ####
fold1_train = Map(fold0_train, pascal_cats=get_cats('train', 1))
fold1_5shot_train = Map(fold1_train,k_shot=5)

fold1_5shot_test = Map(fold0_5shot_test, pascal_cats=get_cats('test', 1))
fold1_1shot_test = Map(fold1_5shot_test, k_shot=1)

