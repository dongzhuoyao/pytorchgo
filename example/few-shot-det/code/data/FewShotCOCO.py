
import numpy as np
import random,os
from multiprocessing import Process, Queue, Pool, Lock
import os.path as osp
import sys
import traceback
import copy,cv2,pickle
from pytorchgo.utils import logger
from tqdm import tqdm
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from pytorchgo.utils.constant import PASCAL_CLASS
from pytorchgo.utils.map_util import Map


IS_DEBUG = 0
PASCAL_PATH= '/data2/dataset/VOCdevkit'


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
    def __init__(self, name, img_path, obj_ids, bbox, ids_map=None):
        self.name = name
        self.img_path = img_path
        self.obj_ids = obj_ids
        self.bbox = bbox
        if ids_map is None:
            self.ids_map = dict(zip(obj_ids, np.ones(len(obj_ids))))
        else:
            self.ids_map = ids_map


    def read_img(self):
        #return read_img(self.img_path), simple
        pass




class PASCAL:
    def __init__(self, db_path, dataType, data_split):

        assert dataType == "train" or dataType == "val"
        self.db_path = db_path
        self.name_id_map = dict(zip(PASCAL_CLASS, range(1, 21)))
        self.id_name_map = dict(zip(range(1, 21), PASCAL_CLASS))
        self.dataType = dataType
        self.data_split = data_split

        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')

        self.keep_difficult = False
        self.class_to_ind = dict(
            zip(PASCAL_CLASS, range(1,len(PASCAL_CLASS)+1)))#start from 1!!!!!


    def getCatIds(self, catNms=[]):
        return [self.name_id_map[catNm] for catNm in catNms]

    def get_anns_path(self, read_mode):
        return osp.join("{}_{}_{}_anns.pkl".format(self.dataType, self.data_split,read_mode))

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


    def _get_bbox(self, target, width, height):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt*1.0 / width if i % 2 == 0 else cur_pt*1.0 / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

    def create_anns(self, read_mode):
        logger.info("create_anns from origin...")
        tuple_list = []
        for (year, name) in [('2007', 'trainval'), ('2012', 'trainval')]:
            rootpath = os.path.join(self.db_path, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                tuple_list.append((rootpath, line.strip()))

        anns = []
        for item in tqdm(tuple_list,total=len(tuple_list),desc="create annotations {}".format(self.data_split)):#per image
            image_root = item[0]
            img_id = item[1]
            class_bbox_dict = {}
            if read_mode == PASCAL_READ_MODES.INSTANCE:
                target = ET.parse(self._annopath % item).getroot()
                img = cv2.imread(self._imgpath % item)
                height, width, channels = img.shape
                target_bboxs = self._get_bbox(target, width, height)

            for bbox in target_bboxs:
                    class_id = bbox[-1]
                    xywh = bbox[:-1]
                    if class_id in class_bbox_dict:
                        class_bbox_dict[class_id]['bbox'].append(xywh)
                    else:
                        class_bbox_dict[class_id] = dict(image_name = item, mask_name = item, class_ids = [class_id], bbox = [xywh])

            anns.extend([value for value in class_bbox_dict.values()])

        return anns


    def load_anns(self, read_mode):
        anns = self.create_anns(read_mode)
        return anns

    def get_anns(self, catIds=[], areaRng=[], read_mode=PASCAL_READ_MODES.INSTANCE):
        if areaRng == []:
            areaRng = [0, np.inf]
        anns = self.load_anns(read_mode)
        if catIds == [] and areaRng == [0, np.inf]:
            return anns

        filtered_anns = [ann for ann in tqdm(anns,total=len(anns)) if
                         ann['class_ids'][0] in catIds]
        print("origin anns={}, filtered anns={}.".format(len(anns),len(filtered_anns)))
        return filtered_anns

    def getItems(self, cats=[], areaRng=[], read_mode=PASCAL_READ_MODES.INSTANCE):
        if len(cats) == 0:
            catIds = self.id_name_map.keys()
        else:
            catIds = self.getCatIds(catNms=cats)
        catIds = np.sort(catIds)

        anns = self.get_anns(catIds=catIds, areaRng=areaRng, read_mode=read_mode)#heavy operation!
        logger.info('{} annotations read from pascal'.format(len(anns)))

        items = []

        for i, ann in enumerate(anns):
            cur_root_path, image_id = ann['image_name']
            img_path = osp.join(cur_root_path, 'JPEGImages',  '{}.jpg'.format(image_id))
            item = DBPascalItem('pascal_{}_path{}_imageid{}_instance{}'.format(self.dataType, cur_root_path, image_id, i),
                                img_path,
                                ann['class_ids'], bbox=ann['bbox'])
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
            if item_id in clusters:
                clusters[item_id].append(item)
            else:
                clusters[item_id] = DBImageSetItem('set class id = {}'.format(item_id), [item])
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
        pickle_path = osp.join("final_clustered_{}.pkl".format(self.params['data_split']))
        if not osp.exists(pickle_path) or IS_DEBUG:
            pascal_db = PASCAL(self.params['pascal_path'], self.params['image_sets'],
                               data_split=self.params['data_split'])  # train or test
            # reads pair of images from one semantic class and and with binary labels
            self.db_items = pascal_db.getItems(self.params['pascal_cats'], read_mode=PASCAL_READ_MODES.INSTANCE)

            logger.info('data result: total of {} db items loaded!'.format(len(self.db_items)))
            clusters = PASCAL.cluster_items(self.db_items)

            # db_items will be a list of tuples (set,j) in which set is the set that img_item belongs to and j is the index of img_item in that set
            final_db_items = []  # empty the list !!
            for item in tqdm(self.db_items, desc="random assign", total=len(self.db_items)):
                set_id = item.obj_ids[0]
                imgset = clusters[set_id]
                assert (imgset.length > self.params[
                    'k_shot']), 'class ' + imgset.name + ' has only ' + imgset.length + ' examples.'
                in_set_index = imgset.image_items.index(item)
                final_db_items.append((imgset, in_set_index))  # in_set_index is used for "second_image"
            self.db_items = final_db_items
            logger.info('data result: total of {} classes!'.format(len(clusters)))

            with open(pickle_path, 'wb') as f:
                logger.info("dump pickle file...")
                pickle.dump(self.db_items, f)

        with open(pickle_path, 'rb') as f:
            logger.warn("loading {} data from pickle file....".format(self.params['data_split']))
            self.db_items = pickle.load(f)


        self.orig_db_items = copy.copy(self.db_items)
        self.seq_index = len(self.db_items)
        

    def update_seq_index(self):
        self.seq_index += 1
        if self.seq_index >= len(self.db_items):# reset status when full
            self.db_items = copy.copy(self.orig_db_items)
            self.rand_gen.shuffle(self.db_items)
            self.seq_index = 0
    
    def next_pair(self):
            """
            end_of_cycle = 'db_cycle' in self.params and self.cycle >= self.params['db_cycle']
            if end_of_cycle:
                assert(self.params['db_cycle'] > 0) # full, reset status
                self.cycle = 0
                self.seq_index = len(self.db_items)
                self.init_randget(self.params['read_mode'])
                
            self.cycle += 1
            self.update_seq_index()



            imgset, second_index = self.db_items[self.seq_index] # query image index
            set_indices = list(range(second_index)) + list(range(second_index+1, len(imgset.image_items))) # exclude second_index
            assert(len(set_indices) >= self.params['k_shot'])
            self.rand_gen.shuffle(set_indices)
            first_index = set_indices[:self.params['k_shot']] # support set image indexes(may be multi-shot~)
            class_id = imgset.image_items[second_index].obj_ids[0]
            metadata = {
                'class_id':class_id,
                'class_name':PASCAL_CLASS[class_id-1],
                'second_image_path':imgset.image_items[second_index].img_path,
                        }

            #TODO, draw bbox
            return [imgset.image_items[v].img_path for v in first_index], \
                   [imgset.image_items[v].bbox for v in first_index], \
                   imgset.image_items[second_index].img_path, \
                   imgset.image_items[second_index].bbox, metadata

            """

            pass

############################################################


def get_cats(split, fold, num_folds=4):
    '''
      Returns a list of categories (for training/test) for a given fold number

      Inputs:
        split: specify train/val
        fold : fold number, out of num_folds
        num_folds: Split the set of image classes to how many folds. In BMVC paper, we use 4 folds

    '''
    num_cats = len(PASCAL_CLASS)
    assert(num_cats%num_folds==0)
    val_size = int(num_cats/num_folds)
    assert(fold<num_folds)
    val_set = [ fold*val_size+v for v in range(val_size)]
    train_set = [x for x in range(num_cats) if x not in val_set]
    if split=='train':
        return [PASCAL_CLASS[x] for x in train_set]
    else:
        return [PASCAL_CLASS[x] for x in val_set]


train_default_profile = Map(
                k_shot=1,
                read_mode=None, # Either "Shuffle" (for training) or "Deterministic" (for testing, random seed fixed)
                pascal_cats = PASCAL_CLASS,
                pascal_path = PASCAL_PATH)

_default_profile = Map(
                k_shot=1,
                read_mode=None, # Either "Shuffle" (for training) or "Deterministic" (for testing, random seed fixed)
                pascal_cats = PASCAL_CLASS,
                pascal_path = PASCAL_PATH)


foldall_train = Map(
                    data_split = "foldall_train",
                    k_shot = 1,
                    read_mode='shuffle',
                    image_sets='train',
                    pascal_cats = PASCAL_CLASS,)


foldall_1shot_val = Map(
                         data_split = "foldall_1shot_val",
                         db_cycle = 1000,
                         read_mode='deterministic',
                         image_sets='val',
                         pascal_cats = PASCAL_CLASS,
                         pascal_path = PASCAL_PATH,
                         k_shot=1)

foldall_5shot_val = Map(
                         db_cycle = 1000,
                         read_mode='deterministic',
                         image_sets='val',
                         pascal_cats = PASCAL_CLASS,
                        data_split = "foldall_5shot_val",
                        pascal_path=PASCAL_PATH,
                         k_shot=5)


#### fold 0 ####

# Setting for training (on **training images**)
fold0_1shot_train = Map(
                    data_split = "fold0_1shot_train",
                    k_shot = 1,
                    read_mode='shuffle',
                    image_sets='train',
                    pascal_path=PASCAL_PATH,
                  pascal_cats = get_cats('train',0)
)

fold0_5shot_train = Map( data_split = "fold0_5shot_train",
                    k_shot = 5,
                    read_mode='shuffle',
                    image_sets='train',
                         pascal_path=PASCAL_PATH,
                  pascal_cats = get_cats('train',0))

# Setting for testing on **test images** in unseen image classes (in total 5 classes), 5-shot
fold0_5shot_val = Map(
                        data_split="fold0_5shot_val",
                       db_cycle = 1000,
                       read_mode='deterministic',
                       image_sets='val',
                        pascal_path=PASCAL_PATH,
                       pascal_cats = get_cats('val',0),
                       k_shot=5)

fold0_1shot_val = Map(
                        data_split="fold0_1shot_val",
                       db_cycle = 1000,
                       read_mode='deterministic',
                       image_sets='val',
                       pascal_cats = get_cats('val',0),
                        pascal_path=PASCAL_PATH,
                       k_shot=1)





################

fold1_1shot_train = Map(
                    data_split = "fold1_1shot_train",
                    k_shot = 1,
                    read_mode='shuffle',
                    image_sets='train',
                    pascal_path=PASCAL_PATH,
                  pascal_cats = get_cats('train',1)
)
fold1_1shot_val = Map(
                        data_split="fold1_1shot_val",
                       db_cycle = 1000,
                       read_mode='deterministic',
                       image_sets='val',
                       pascal_cats = get_cats('val',1),
                        pascal_path=PASCAL_PATH,
                       k_shot=1)


################

fold2_1shot_train = Map(
                    data_split = "fold2_1shot_train",
                    k_shot = 1,
                    read_mode='shuffle',
                    image_sets='train',
                    pascal_path=PASCAL_PATH,
                  pascal_cats = get_cats('train',2)
)
fold2_1shot_val = Map(
                        data_split="fold2_1shot_val",
                       db_cycle = 1000,
                       read_mode='deterministic',
                       image_sets='val',
                       pascal_cats = get_cats('val',2),
                        pascal_path=PASCAL_PATH,
                       k_shot=1)

################

fold3_1shot_train = Map(
                    data_split = "fold3_1shot_train",
                    k_shot = 1,
                    read_mode='shuffle',
                    image_sets='train',
                    pascal_path=PASCAL_PATH,
                  pascal_cats = get_cats('train',3)
)
fold3_1shot_val = Map(data_split="fold3_1shot_val",
                       db_cycle = 1000,
                       read_mode='deterministic',
                       image_sets='val',
                       pascal_cats = get_cats('val',3),
                        pascal_path=PASCAL_PATH,
                       k_shot=1)





