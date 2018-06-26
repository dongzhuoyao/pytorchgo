
import numpy as np
import random,os
from multiprocessing import Process, Queue, Pool, Lock
import os.path as osp
import sys
import traceback
import util
from util import cprint, bcolors
from skimage.transform import resize
import copy,cv2,pickle

class PASCAL_READ_MODES:
    #Returns list of DBImageItem each has the image and one object instance in the mask
    INSTANCE = 0
    #Returns list of DBImageItem each has the image and the mask for all semantic labels
    SEMANTIC_ALL = 1
    #Returns list of DBImageSetItem each has set of images and corresponding masks for each semantic label
    SEMANTIC = 2


class DBImageSetItem():
    def __init__(self, name, image_items=[]):
        self.name = name
        self.length = len(image_items)
        self.image_items = image_items

    def append(self, image_item):
        self.image_items.append(image_item)
        self.length += 1

    def read_img(self, img_id):
        return self.image_items[img_id].read_img()

    def read_mask(self, img_id):
        return self.image_items[img_id].read_mask()


class DBPascalItem():
    def __init__(self, name, img_path, mask_path, obj_ids, ids_map=None):
        self.name = name
        self.img_path = img_path
        self.mask_path = mask_path
        self.obj_ids = obj_ids
        if ids_map is None:
            self.ids_map = dict(zip(obj_ids, np.ones(len(obj_ids))))
        else:
            self.ids_map = ids_map

    def read_mask(self, orig_mask=False):
        mobj_uint = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        if orig_mask:
            return mobj_uint.astype(np.float32)
        m = np.zeros(mobj_uint.shape, dtype=np.float32)
        for obj_id in self.obj_ids:
            m[mobj_uint == obj_id] = self.ids_map[obj_id]
        return m

    def read_img(self):
        #return read_img(self.img_path), simple
        pass




class PASCAL:
    def __init__(self, db_path, dataType):

        assert dataType == "train" or dataType == "val"
        self.db_path = db_path
        classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
        self.name_id_map = dict(zip(classes, range(1, len(classes) + 1)))
        self.id_name_map = dict(zip(range(1, len(classes) + 1), classes))
        self.dataType = dataType

    def getCatIds(self, catNms=[]):
        return [self.name_id_map[catNm] for catNm in catNms]

    def get_anns_path(self, read_mode):
        return osp.join(self.db_path, "{}_{}_anns.pkl".format(self.dataType, str(read_mode)))

    def get_unique_ids(self, mask, return_counts=False, exclude_ids=[0, 255]):
        ids, sizes = np.unique(mask, return_counts=True)
        ids = list(ids)
        sizes = list(sizes)
        for ex_id in exclude_ids:
            if ex_id in ids:
                id_index = ids.index(ex_id)
                ids.remove(ex_id)
                sizes.remove(sizes[id_index])

        assert (len(ids) == len(sizes))

        if return_counts:
            return ids, sizes
        else:
            return ids

    def create_anns(self, read_mode):
        list_dir = 'metadata/pascalvoc12/train_aug_id.txt'
        if self.dataType == "val":
            list_dir = 'metadata/pascalvoc12/val_id.txt'
        with open(list_dir, 'r') as f:
            lines = f.readlines()
            names = []
            for line in lines:
                if line.endswith('\n'):
                    line = line[:-1]
                if len(line) > 0:
                    names.append(line)
        anns = []
        for item in names:
            mclass_path = osp.join(self.db_path, 'SegmentationClassAug', item + '.png')
            mclass_uint = cv2.imread(mclass_path, cv2.IMREAD_GRAYSCALE)
            class_ids = self.get_unique_ids(mclass_uint)

            if read_mode == PASCAL_READ_MODES.SEMANTIC:
                for class_id in class_ids:
                    assert (class_id != 0 or class_id != 255)  # 0 is background,255 is ignore
                    anns.append(dict(image_name=item, mask_name=item, class_ids=[class_id]))
            elif read_mode == PASCAL_READ_MODES.SEMANTIC_ALL:
                anns.append(dict(image_name=item, mask_name=item, class_ids=class_ids))
            else:
                raise ValueError
        with open(self.get_anns_path(read_mode), 'w') as f:
            pickle.dump(anns, f)

    def load_anns(self, read_mode):
        path = self.get_anns_path(read_mode)
        if not osp.exists(path):
            self.create_anns(read_mode)
        with open(path, 'rb') as f:
            anns = pickle.load(f)
        return anns

    def get_anns(self, catIds=[], areaRng=[], read_mode=PASCAL_READ_MODES.INSTANCE):
        if areaRng == []:
            areaRng = [0, np.inf]
        anns = self.load_anns(read_mode)
        if catIds == [] and areaRng == [0, np.inf]:
            return anns

        if read_mode == PASCAL_READ_MODES.INSTANCE:
            filtered_anns = [ann for ann in anns if
                             ann['class_ids'][0] in catIds and areaRng[0] < ann['object_sizes'][0] and
                             ann['object_sizes'][0] < areaRng[1]]
        else:
            filtered_anns = []
            catIds_set = set(catIds)
            for ann in anns:
                class_inter = set(ann['class_ids']) & catIds_set
                # remove class_ids that we did not asked for (i.e. are not catIds_set)
                if len(class_inter) > 0:
                    ann = ann.copy()
                    ann['class_ids'] = sorted(list(class_inter))
                    filtered_anns.append(ann)
        return filtered_anns

    def getItems(self, cats=[], areaRng=[], read_mode=PASCAL_READ_MODES.INSTANCE):
        if len(cats) == 0:
            catIds = self.id_name_map.keys()
        else:
            catIds = self.getCatIds(catNms=cats)
        catIds = np.sort(catIds)

        anns = self.get_anns(catIds=catIds, areaRng=areaRng, read_mode=read_mode)
        cprint(str(len(anns)) + ' annotations read from pascal', bcolors.OKGREEN)

        items = []

        ids_map = None
        if read_mode == PASCAL_READ_MODES.SEMANTIC_ALL:
            old_ids = catIds
            new_ids = range(1, len(catIds) + 1)
            ids_map = dict(zip(old_ids, new_ids))
        for i in range(len(anns)):
            ann = anns[i]
            img_path = osp.join(self.db_path, 'JPEGImages', ann['image_name'] + '.jpg')
            mask_path = osp.join(self.db_path, 'SegmentationClassAug', ann['mask_name'] + '.png')
            item = DBPascalItem('pascal-' + self.dataType + '_' + ann['image_name'] + '_' + str(i), img_path, mask_path,
                                ann['class_ids'], ids_map)
            items.append(item)
        return items

    @staticmethod
    def cluster_items(items):
        clusters = {}
        for i, item in enumerate(items):
            assert (isinstance(item, DBPascalItem))
            item_id = item.obj_ids
            assert (len(item_id) == 1), 'For proper clustering, items should only have one id'
            item_id = item_id[0]
            if clusters.has_key(item_id):
                clusters[item_id].append(item)
            else:
                clusters[item_id] = DBImageSetItem('set class id = ' + str(item_id), [item])
        return clusters


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
            

