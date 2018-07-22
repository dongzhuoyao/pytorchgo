import os
import sys
import cv2
import argparse
import numpy as np

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import settings
from dataset import TrainValDataset, TestDataset
from model import DetailNet
from cal_ssim import SSIM

logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        

class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        self.net = DetailNet().cuda()
        self.crit = MSELoss().cuda()
        self.ssim = SSIM().cuda()

        self.step = 0
        self.save_steps = settings.save_steps
        self.num_workers = settings.num_workers
        self.batch_size = settings.batch_size
        self.writers = {}
        self.dataloaders = {}

        self.opt = Adam(self.net.parameters(), lr=settings.lr)
        self.sche = MultiStepLR(self.opt, milestones=[15000, 17500], gamma=0.1)

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)

        out['lr'] = self.opt.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v) 
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def get_dataloader(self, dataset_name, train_mode=True):
        dataset = {
            True: TrainValDataset,
            False: TestDataset,
        }[train_mode](dataset_name)
        self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=self.batch_size, 
                            shuffle=True, num_workers=self.num_workers, drop_last=True)
        if train_mode:
            return iter(self.dataloaders[dataset_name])
        else:
            return self.dataloaders[dataset_name]

    def save_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.net.state_dict(),
            'clock': self.step,
            'opt': self.opt.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            return
        self.net.load_state_dict(obj['net'])
        self.opt.load_state_dict(obj['opt'])
        self.step = obj['clock']
        self.sche.last_epoch = self.step

    def inf_batch(self, name, batch):
        O, B = batch['O'], batch['B']
        O, B = O.cuda(), B.cuda()
        O, B = Variable(O), Variable(B)
        R = O - B

        O_Rs = self.net(O)
        loss_list = [self.crit(O_R, R) for O_R in O_Rs]
        ssim_list = [self.ssim(O - O_R, O - R) for O_R in O_Rs]

        loss = sum(loss_list)
        losses = {
            'loss%d' % i: loss.data[0]
            for i, loss in enumerate(loss_list)
        }
        ssimes = {
            'ssim%d' % i: ssim.data[0]
            for i, ssim in enumerate(ssim_list)
        }
        losses.update(ssimes)

        return O - O_Rs[-1], loss, losses

    def save_image(self, name, img_lists):
        data, pred, label = img_lists
        pred = pred.cpu().data

        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)

        h, w = pred.shape[-2:]

        gen_num = (6, 2)
        img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx]]
                    for k in range(3):
                        col = (j * 3 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row+h, col: col+w] = tmp 

        img_file = os.path.join(self.log_dir, '%d_%s.jpg' % (self.step, name))
        cv2.imwrite(img_file, img)


def run_train_val(ckp_name='latest'):
    sess = Session()
    sess.load_checkpoints(ckp_name)

    sess.tensorboard('train')
    sess.tensorboard('val')

    dt_train = sess.get_dataloader('train')
    dt_val = sess.get_dataloader('val')

    while sess.step < 20000:
        sess.sche.step()

        sess.net.train()
        sess.net.zero_grad()

        batch_t = next(dt_train)
        pred_t, loss_t, losses_t = sess.inf_batch('train', batch_t)
        sess.write('train', losses_t)
        loss_t.backward()
        sess.opt.step()

        if sess.step % 4 == 0:
            sess.net.eval()
            batch_v = next(dt_val)
            pred_v, loss_v, losses_v = sess.inf_batch('val', batch_v)
            sess.write('val', losses_v)

        if sess.step % int(sess.save_steps / 16) == 0:
            sess.save_checkpoints('latest')
        if sess.step % int(sess.save_steps / 2) == 0:
            sess.save_image('train', [batch_t['O'], pred_t, batch_t['B']])
            if sess.step % 4 == 0:
                sess.save_image('val', [batch_v['O'], pred_v, batch_v['B']])
            logger.info('save image as step_%d' % sess.step)
        if sess.step % sess.save_steps == 0:
            sess.save_checkpoints('step_%d' % sess.step)
            logger.info('save model as step_%d' % sess.step)
        sess.step += 1


def run_test(ckp_name):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints(ckp_name)

    dt = sess.get_dataloader('test', train_mode=False)

    all_num = 0
    all_losses = {}
    for i, batch in enumerate(dt):
        pred, loss, losses = sess.inf_batch('test', batch)
        batch_size = pred.size(0)
        all_num += batch_size
        for key, val in losses.items():
            if i == 0:
                all_losses[key] = 0.
            all_losses[key] += val * batch_size
            logger.info('batch %d mse %s: %f' % (i, key, val))

    for key, val in all_losses.items():
        logger.info('total mse %s: %f' % (key, val / all_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', default='train')
    parser.add_argument('-m', '--model', default='latest')

    args = parser.parse_args(sys.argv[1:])
    
    if args.action == 'train':
        run_train_val(args.model)
    elif args.action == 'test':
        run_test(args.model)

