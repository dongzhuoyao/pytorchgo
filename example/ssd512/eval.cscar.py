"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from pytorchgo.dataloader.cs_car_loader import CsCar_CLASSES as labelmap
import torch.utils.data as data
from tqdm import tqdm
from data import  BaseTransform
from ssd import build_ssd
from log import log
import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
from pytorchgo.utils import logger
is_debug = 0

txt_path = "/home/hutao/lab/pytorchgo/dataset_list/cityscapes_car/car_val.txt"
sim_path = '/home/hutao/dataset/cityscapes'
restore_from = '/home/hutao/lab/pytorchgo/example/ssd512/train_log/train.cs_car.512/ssd_39999.pth'
dataset_mean = (104, 117, 123)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default=restore_from,
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--sim_root', default=sim_path, help='Location of VOC root directory')

args = parser.parse_args()
logger.auto_set_dir()

per_class_result_path = os.path.join(logger.get_logger_dir(),"{}_det_result.txt")


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'


if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(annos):
    detect_obj = annos.split(";")
    objects = []
    for obj in detect_obj:
        xmin = int(obj.split(",")[0])
        ymin = int(obj.split(",")[1])
        xmax = int(obj.split(",")[2])
        ymax = int(obj.split(",")[3])
        obj_struct = {}
        obj_struct['name'] = 'car'
        obj_struct['difficult'] = 0  # all is not difficult!
        obj_struct['bbox'] = [xmin,ymin, xmax, ymax]
        objects.append(obj_struct)

    return objects


def write_class_results_file(all_boxes, dataset):#each boundingbox a line
    for cls_ind, cls in enumerate(labelmap):
        log.l.info('Writing {:s} class det results file'.format(cls))
        with open(per_class_result_path.format(cls), 'wt') as f:
            for im_ind, index in enumerate(dataset.files):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    cur_line = '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(index.split(" ")[0], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1)# image, score, minx,miny,maxx,maxy
                    f.write(cur_line)


def do_python_eval( use_07=True):
    cachedir = os.path.join(logger.get_logger_dir(), 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    log.l.info('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

    for i, cls in enumerate(labelmap):
        rec, prec, ap = voc_eval(
            cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        log.l.info('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(logger.get_logger_dir(), cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    log.l.info('Mean AP = {:.4f}'.format(np.mean(aps)))
    log.l.info('~~~~~~~~')
    log.l.info('Results:')
    for ap in aps:
        log.l.info('{:.3f}'.format(ap))
    log.l.info('{:.3f}'.format(np.mean(aps)))
    log.l.info('~~~~~~~~')
    log.l.info('')
    log.l.info('--------------------------------------------------------------')
    log.l.info('Results computed with the **unofficial** Python eval code.')
    log.l.info('Results should be very close to the official MATLAB eval code.')
    log.l.info('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default False)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(txt_path, 'r') as f:
        image_dict = {x.strip().split(" ")[0]:x.strip().split(" ")[1] for x in f.readlines()}

    if True:#not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, (imagename, annos) in enumerate(image_dict.items()):
            recs[imagename] = parse_rec(annos)
            if i % 100 == 0:
                log.l.info('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(image_dict)))
        # save
        log.l.info('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in image_dict:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = per_class_result_path.format(classname)
    with open(detfile, 'r') as f:
        det_bd_lines = f.readlines()
    if any(det_bd_lines) == 1:

        splitlines = [x.strip().split(' ') for x in det_bd_lines]
        image_ids = [x[0] for x in splitlines] # this may include dunplicat image_id, because each image may include many boundingboxes.
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids) # bound_box number
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(net, dataset):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]#plus the background, [class_num, image_num]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    for i in tqdm(range(num_images),total=num_images):
        if i > 100 and is_debug==1:break


        im, gt, h, w = dataset.__getitem__(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data #[1, 2, 200, 5]
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):# this is by default!
            dets = detections[0, j, :]#[200, 5]

            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t() #[200, 5]
            dets = torch.masked_select(dets, mask).view(-1, 5) #[129, 5]

            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False) #[129, 5]
            all_boxes[j][i] = cls_dets

        log.l.info('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(os.path.join(logger.get_logger_dir(), 'detection_result.pkl'), 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    log.l.info('Evaluating detections')
    evaluate_detections(all_boxes, dataset)


def evaluate_detections(box_list, dataset):
    write_class_results_file(box_list, dataset)
    do_python_eval()


if __name__ == '__main__':
    # load net
    num_classes = 2
    image_size = 512
    net = build_ssd('test', image_size, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model,map_location={'cuda:2':'cuda:1'}))
    net.eval()
    log.l.info('Finished loading model!')
    # load data
    from pytorchgo.dataloader.cs_car_loader import CsCarDetection
    dataset = CsCarDetection(split="val", transform=BaseTransform(image_size, dataset_mean))
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net( net, dataset)
