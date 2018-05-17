import datetime
import math
import os
import os.path as osp
import shutil
import fcn
import numpy as np
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import itertools, datetime
import torchvision.utils as vutils
import torchfcn
import torch.nn as nn
from util_fns import get_parameters
from utils import cross_entropy2d, step_scheduler
from pytorchgo.loss import MSE_Loss
from pytorchgo.function import grad_reverse
from pytorchgo.utils import logger

CLASS_NUM = 19


class MyTrainer_ROAD(object):
    def __init__(self, cuda, model, model_fix, netD, optimizer, optimizerD,
                 train_loader, target_loader, val_loader,
                 max_iter, image_size,batch_size,
                 size_average=True, interval_validate=None):
        self.cuda = cuda
        self.model = model
        self.model_fix = model_fix
        self.netD = netD
        self.optim = optimizer
        self.optimD = optimizerD
        self.batch_size = batch_size

        self.train_loader = train_loader
        self.target_loader = target_loader
        self.val_loader = val_loader

        self.image_size = tuple(image_size)
        self.n_class = len(self.train_loader.dataset.class_names)

        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = min(len(self.train_loader), len(self.target_loader))
        else:
            self.interval_validate = interval_validate

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0

    def validate(self):
        """
        Function to validate a training model on the val split.
        """
        logger.info("start validation....")
        self.model.eval()

        val_loss = 0
        label_trues, label_preds = [], []

        # Evaluation
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Validation iteration = %d' % self.iteration, ncols=80,
                leave=False):

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            score = self.model(data)

            loss = cross_entropy2d(score, target, size_average=self.size_average)
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while validating')
            val_loss += float(loss.data[0]) / len(data)

            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()

            label_trues.append(lbl_true)
            label_preds.append(lbl_pred)

        # Computing the metrics
        acc, acc_cls, mean_iu, _ = torchfcn.utils.label_accuracy_score(
            label_trues, label_preds, self.n_class)
        val_loss /= len(self.val_loader)

        logger.info("validation mIoU = {}".format(mean_iu))

        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(logger.get_logger_dir(), 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(logger.get_logger_dir(), 'checkpoint.pth.tar'),
                        osp.join(logger.get_logger_dir(), 'model_best.pth.tar'))

    def train_epoch(self):
        """
        Function to train the model for one epoch
        """
        self.model.train()
        self.netD.train()


        for batch_idx, (datas, datat) in tqdm.tqdm(
                enumerate(itertools.izip(self.train_loader, self.target_loader)),
                total=self.iters_per_epoch,
                desc='Train epoch = %d' % self.epoch, ncols=80, leave=False):

            source_data, source_labels = datas
            target_data, __ = datat



            #continue
            # source =0, target = 1
            source_data_forD = torch.zeros((self.batch_size, 3, self.image_size[1], self.image_size[0]))
            target_data_forD = torch.zeros((self.batch_size, 3, self.image_size[1], self.image_size[0]))

            # We pass the unnormalized data to the discriminator. So, the GANs produce images without data normalization
            for i in range(self.batch_size):
                source_data_forD[i] = self.train_loader.dataset.transform_forD(source_data[i], self.image_size,
                                                                               resize=False, mean_add=True)
                target_data_forD[i] = self.train_loader.dataset.transform_forD(target_data[i], self.image_size,
                                                                               resize=False, mean_add=True)

            self.iteration = batch_idx + self.epoch * self.iters_per_epoch


            if self.cuda:
                source_data, source_labels = source_data.cuda(), source_labels.cuda()
                target_data = target_data.cuda()

            source_data, source_labels = Variable(source_data), Variable(source_labels)
            target_data = Variable(target_data)

            # TODO,split to 3x3
            # Source domain
            score = self.model(source_data)
            l_seg = cross_entropy2d(score, source_labels, size_average=self.size_average)

            # outD_src_real_s, outD_src_real_c = self.netD(score)

            # target domain
            seg_target_score = self.model(target_data)
            modelfix_target_score = self.model_fix(target_data)
            # outD_tgt_fake_s, outD_tgt_fake_c = self.netD(tscore)


            distill_loss = MSE_Loss(seg_target_score, modelfix_target_score)

            self.optim.zero_grad()
            total_loss = l_seg + 0.1 * distill_loss
            total_loss.backward(retain_graph=True)
            self.optim.step()

            if self.iteration % 1 == 0:
                logger.info("L_SEG={}, Distill_LOSS={}, TOTAL_LOSS :{}".format(l_seg.data[0], distill_loss.data[0],
                                                                               total_loss.data[0]))

            no_0_0 = 1

            # TODO, GRL layer
            # grad_reverse()

            # TODO, spatial loss

            if np.isnan(float(total_loss.data[0])):
                raise ValueError('total_loss is nan while training')

            if self.iteration >= self.max_iter:
                break

            # Validating periodically
            if self.iteration % self.interval_validate == 0 and self.iteration > 0:
                self.validate()
                self.model.train()  # return to training mode

    def train(self):
        """
        Function to train our model. Calls train_epoch function every epoch.
        Also performs learning rate annhealing
        """
        logger.info("train_loader length: {}".format(len(self.train_loader)))
        logger.info("target_loader length: {}".format(len(self.target_loader)))
        iters_per_epoch = min(len(self.target_loader), len(self.train_loader))

        self.iters_per_epoch = iters_per_epoch - (iters_per_epoch % self.batch_size) - 1

        logger.info("iters_per_epoch :{}".format(self.iters_per_epoch))
        max_epoch = int(math.ceil(self.max_iter / self.iters_per_epoch))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            if self.epoch % 8 == 0 and self.epoch > 0:
                self.optim = step_scheduler(self.optim, self.epoch)


            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
