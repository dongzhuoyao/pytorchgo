
import numpy as np
import random,os
from multiprocessing import Process, Queue, Pool, Lock

import sys
import traceback
import util
from util import cprint, bcolors
from skimage.transform import resize
import copy


class DBInterface():
    def __init__(self, params):
        self.params = params
        self.load_items()
        self.data_size = len(self.db_items)
        # initialize the random generator
        self.init_randget(params['read_mode'])
        self.cycle = 0

    def init_randget(self, read_mode):
        self.rand_gen = random.Random()
        if read_mode == 'shuffle':
            self.rand_gen.seed()
        elif read_mode == 'deterministic':
            self.rand_gen.seed(1385)  # >>>Do not change<<< Fixed seed for deterministic mode.

    def load_items(self):
        def _remove_small_objects(items):
            filtered_item = []
            for item in items:
                mask = item.read_mask()
                if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
                    filtered_item.append(item)
            return filtered_item

        self.db_items = []
        if self.params.has_key('image_sets'):
            for image_set in self.params['image_sets']:
                pascal_db = util.PASCAL(self.params['pascal_path'], image_set.replace("pascal_",""))  # train or test
                # reads pair of images from one semantic class and and with binary labels
                items = pascal_db.getItems(self.params['pascal_cats'], self.params['areaRng'],
                                           read_mode=util.PASCAL_READ_MODES.SEMANTIC)
                items = _remove_small_objects(items)
                self.db_items.extend(items)

            cprint('Total of ' + str(len(self.db_items)) + ' db items loaded!', bcolors.OKBLUE)
            # In image_pair mode pair of images are sampled from the same semantic class
            clusters = util.PASCAL.cluster_items(self.db_items)

            # db_items will be a list of tuples (set,j) in which set is the set that img_item belongs to and j is the index of img_item in that set
            self.db_items = []  # empty the list !!
            for item in self.db_items:
                set_id = item.obj_ids[0]
                imgset = clusters[set_id]
                assert (imgset.length > self.params[
                    'k_shot']), 'class ' + imgset.name + ' has only ' + imgset.length + ' examples.'
                in_set_index = imgset.image_items.index(item)
                self.db_items.append((imgset, in_set_index)) #in_set_index is used for "second_image"
            cprint('Total of ' + str(len(clusters)) + ' classes!', bcolors.OKBLUE)

        self.orig_db_items = copy.copy(self.db_items)
        self.seq_index = len(self.db_items)
        

    def update_seq_index(self):
        self.seq_index += 1
        if self.seq_index >= len(self.db_items):# reset status when full
            self.db_items = copy.copy(self.orig_db_items)
            self.rand_gen.shuffle(self.db_items)
            self.seq_index = 0
    
    def next_pair(self):
            end_of_cycle = self.params.has_key('db_cycle') and self.cycle >= self.params['db_cycle']
            if end_of_cycle:
                assert(self.params['db_cycle'] > 0) # full, reset status
                self.cycle = 0
                self.seq_index = len(self.db_items)
                self.init_randget(self.params['read_mode'])
                
            self.cycle += 1
            self.update_seq_index()

            imgset, second_index = self.db_items[self.seq_index] # query image index
            set_indices = range(second_index) + range(second_index+1, len(imgset.image_items)) # exclude second_index
            assert(len(set_indices) >= self.params['k_shot'])
            self.rand_gen.shuffle(set_indices)
            first_index = set_indices[:self.params['k_shot']] # support set image indexes(may be multi-shot~)

            metadata = {'name':imgset.name,
                        'class_id':imgset.image_items[0].obj_ids[0],
                        'image1_name':[os.path.basename(imgset.image_items[ii].img_path) for ii in first_index],
                        'image2_name': os.path.basename(imgset.image_items[second_index].img_path),
                        }

            return [imgset.image_items[v].img_path for v in first_index],\
                   [imgset.image_items[v].mask_path for v in first_index],\
                   imgset.image_items[second_index].img_path,\
                    imgset.image_items[second_index].mask_path, \
                    metadata




            

class PairLoaderProcess():
    def __init__(self, db_interface, params):
        self.db_interface = db_interface
        self.first_shape = params['first_shape']
        self.second_shape = params['second_shape']

        self.deploy_mode = params['deploy_mode'] if params.has_key('deploy_mode') else False
            

    def load_next_frame(self):
        return self.db_interface.next_pair()


    def read_imgs(self, player, first_index, second_index):
        pass
            

