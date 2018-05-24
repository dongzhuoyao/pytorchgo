import torch
import argparse
from util_fns import weights_init
import math
import os
import os.path as osp
import shutil
import fcn
import numpy as np
import pytz
import scipy.misc
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import itertools,datetime
import torch.nn as nn
import torchfcn
from torchfcn.utils import cross_entropy2d, step_scheduler
from pytorchgo.utils import logger

class_num = 16


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        torchfcn.models.FCN8s_sourceonly,
        torchfcn.models.FCN8s_LSD,
        torchfcn.models.Res_Deeplab
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))

def main():
    logger.auto_set_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/home/hutao/lab/pytorchgo/example/LSD-seg/data',
                        help='Path to source dataset')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--num_iters', type=int, default=100000, help='Number of training iterations')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use | SGD, Adam')
    parser.add_argument('--lr', type=float, default=1.0e-5, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.99, help='Momentum for SGD')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--interval_validate', type=int, default=500,
                        help='Period for validation. Model is validated every interval_validate iterations')
    parser.add_argument('--resume', default='',
                        help="path to the current checkpoint for resuming training. Do not specify if model has to be trained from scratch")
    parser.add_argument('--method', default='LSD', help="Method to use for training | LSD, sourceonly")
    parser.add_argument('--l1_weight', type=float, default=1, help='L1 weight')
    parser.add_argument('--adv_weight', type=float, default=0.1, help='Adv_weight')
    parser.add_argument('--c_weight', type=float, default=0.1, help='C_weight')
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()
    print(args)

    gpu = args.gpu
    out = logger.get_logger_dir()
    resume = args.resume
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # Defining data loaders

    image_size = [640, 320]
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SYNTHIA('SYNTHIA', args.dataroot, split='train', transform=True, image_size=image_size, class_num=class_num),
        batch_size=args.batchSize, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SYNTHIA('SYNTHIA', args.dataroot, split='val', transform=True, image_size=image_size, class_num=class_num),
        batch_size=args.batchSize, shuffle=False, **kwargs)
    target_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.CityScapes('cityscapes', args.dataroot, split='train', transform=True, image_size=image_size, class_num=class_num),
        batch_size=args.batchSize, shuffle=True)

    # Defining models

    start_epoch = 0
    start_iteration = 0
    if args.method == 'sourceonly':
        model = torchfcn.models.FCN8s_sourceonly(n_class=class_num)
    elif args.method == 'LSD':
        model = torchfcn.models.FCN8s_LSD(n_class=class_num)
        netG = torchfcn.models._netG()
        netD = torchfcn.models._netD()
        netD.apply(weights_init)
        netG.apply(weights_init)
    else:
        raise ValueError('method argument can be either sourceonly or LSD')

    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()
        if args.method == 'LSD':
            netD = netD.cuda()
            netG = netG.cuda()

    # Defining optimizer

    if args.optimizer == 'SGD':
        optim = torch.optim.SGD(
            [
                {'params': get_parameters(model, bias=False)},
                {'params': get_parameters(model, bias=True),
                 'lr': args.lr * 2, 'weight_decay': args.weight_decay},
            ],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optim = torch.optim.Adam(
            [
                {'params': get_parameters(model, bias=False)},
                {'params': get_parameters(model, bias=True),
                 'lr': args.lr * 2},
            ],
            lr=args.lr,
            betas=(args.beta1, 0.999))
    else:
        raise ValueError('Invalid optmizer argument. Has to be SGD or Adam')

    if args.method == 'LSD':
        optimD = torch.optim.Adam(netD.parameters(), lr=0.0001, betas=(0.7, 0.999))
        optimG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.7, 0.999))

    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    # Defining trainer object, and start training
    if args.method == 'sourceonly':
        trainer = torchfcn.Trainer_sourceonly(
            cuda=cuda,
            model=model,
            optimizer=optim,
            train_loader=train_loader,
            target_loader=target_loader,
            val_loader=val_loader,
            out=out,
            max_iter=args.num_iters,
            interval_validate=args.interval_validate
        )
        trainer.epoch = start_epoch
        trainer.iteration = start_iteration
        trainer.train()
    elif args.method == 'LSD':
        trainer = Trainer_LSD(
            cuda=cuda,
            model=model,
            netD=netD,
            netG=netG,
            optimizer=optim,
            optimizerD=optimD,
            optimizerG=optimG,
            train_loader=train_loader,
            target_loader=target_loader,
            l1_weight=args.l1_weight,
            adv_weight=args.adv_weight,
            c_weight=args.c_weight,
            val_loader=val_loader,
            out=out,
            max_iter=args.num_iters,
            interval_validate=args.interval_validate,
            image_size=image_size
        )
        trainer.epoch = start_epoch
        trainer.iteration = start_iteration
        trainer.train()

class Trainer_LSD(object):
    def __init__(self, cuda, model, netD, netG, optimizer, optimizerD,
                 optimizerG, train_loader, target_loader, val_loader,
                 out, max_iter, l1_weight, adv_weight, image_size, c_weight=0.1,
                 size_average=False, interval_validate=None):
        self.cuda = cuda
        self.model = model
        self.netD = netD
        self.netG = netG
        self.optim = optimizer
        self.optimD = optimizerD
        self.optimG = optimizerG

        self.adv_weight = adv_weight
        self.l1_weight = l1_weight
        self.c_weight = c_weight

        self.train_loader = train_loader
        self.target_loader = target_loader
        self.val_loader = val_loader

        self.lrd = 0.0002
        self.image_size_forD = tuple(image_size)
        self.n_class = len(self.train_loader.dataset.class_names)

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = min(len(self.train_loader), len(self.target_loader))
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'loss',
            'acc',
            'acc_cls',
            'mean_iu',
            'fwavacc',
            'elapsed_time',
        ]

        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0

    def validate(self):
        """
        Function to validate a training model on the val split.
        """

        self.model.eval()
        self.netG.eval()

        val_loss = 0
        num_vis = 8
        visualizations = []
        generations = []
        label_trues, label_preds = [], []

        # Evaluation
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Validation iteration = %d' % self.iteration):

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            score, fc7, pool4, pool3 = self.model(data)
            outG = self.netG(fc7, pool4, pool3)

            loss = cross_entropy2d(score, target, size_average=self.size_average)
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while validating')
            val_loss += float(loss.data[0]) / len(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()

            # Visualizing predicted labels
            for img, lt, lp, outG_ in zip(imgs, lbl_true, lbl_pred, outG):

                outG_ = outG_ * 255.0
                outG_ = outG_.data.cpu().numpy().squeeze().transpose((1, 2, 0))[:, :, ::-1].astype(np.uint8)
                img = self.val_loader.dataset.untransform(img.numpy())
                lt = lt.numpy()
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < num_vis:
                    lt[lt >= CLASS_NUM] = -1  # to make fcn.utils.visualize_segmentation work!
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=self.n_class)
                    visualizations.append(viz)
                    generations.append(outG_)

        # Computing the metrics
        metrics = torchfcn.utils.label_accuracy_score(
            label_trues, label_preds, self.n_class)
        val_loss /= len(self.val_loader)

        # Saving the label visualizations and generations
        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d_labelmap.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))
        out_file = osp.join(out, 'iter%012d_generations.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(generations))

        # Logging
        logger.info("validation mIoU: {}".format(metrics[2]))
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = \
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - \
                self.timestamp_start
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        # Saving the models
        mean_iu = metrics[2]
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
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

    def train_epoch(self):
        """
        Function to train the model for one epoch
        """
        self.model.train()
        self.netG.train()
        self.netD.train()

        for batch_idx, (datas, datat) in tqdm.tqdm(
                enumerate(itertools.izip(self.train_loader, self.target_loader)),
                total=min(len(self.target_loader), len(self.train_loader)),
                desc='Train epoch = {}/{}'.format(self.epoch, self.max_epoch)):

            data_source, labels_source = datas
            data_target, __ = datat
            data_source_forD = torch.zeros((data_source.size()[0], 3, self.image_size_forD[1], self.image_size_forD[0]))
            data_target_forD = torch.zeros((data_target.size()[0], 3, self.image_size_forD[1], self.image_size_forD[0]))

            # We pass the unnormalized data to the discriminator. So, the GANs produce images without data normalization
            for i in range(data_source.size()[0]):
                data_source_forD[i] = self.train_loader.dataset.transform_forD(data_source[i], self.image_size_forD,
                                                                               resize=False, mean_add=True)
                data_target_forD[i] = self.train_loader.dataset.transform_forD(data_target[i], self.image_size_forD,
                                                                               resize=False, mean_add=True)

            iteration = batch_idx + self.epoch * min(len(self.train_loader), len(self.target_loader))
            self.iteration = iteration

            if self.cuda:
                data_source, labels_source = data_source.cuda(), labels_source.cuda()
                data_target = data_target.cuda()
                data_source_forD = data_source_forD.cuda()
                data_target_forD = data_target_forD.cuda()

            data_source, labels_source = Variable(data_source), Variable(labels_source)
            data_target = Variable(data_target)
            data_source_forD = Variable(data_source_forD)
            data_target_forD = Variable(data_target_forD)

            # Source domain
            score, fc7, pool4, pool3 = self.model(data_source)
            outG_src = self.netG(fc7, pool4, pool3)
            outD_src_fake_s, outD_src_fake_c = self.netD(outG_src)
            outD_src_real_s, outD_src_real_c = self.netD(data_source_forD)

            # target domain
            tscore, tfc7, tpool4, tpool3 = self.model(data_target)
            outG_tgt = self.netG(tfc7, tpool4, tpool3)
            outD_tgt_real_s, outD_tgt_real_c = self.netD(data_target_forD)
            outD_tgt_fake_s, outD_tgt_fake_c = self.netD(outG_tgt)

            # Creating labels for D. We need two sets of labels since our model is a ACGAN style framework.
            # (1) Labels for the classsifier branch. This will be a downsampled version of original segmentation labels
            # (2) Domain lables for classifying source real, source fake, target real and target fake

            # Labels for classifier branch
            Dout_sz = outD_src_real_s.size()
            label_forD = torch.zeros(
                (outD_tgt_fake_c.size()[0], outD_tgt_fake_c.size()[2], outD_tgt_fake_c.size()[3]))  # [1,40,80]
            for i in range(label_forD.size()[0]):
                label_forD[i] = self.train_loader.dataset.transform_label_forD(labels_source[i], (
                outD_tgt_fake_c.size()[2], outD_tgt_fake_c.size()[3]))
            if self.cuda:
                label_forD = label_forD.cuda()
            label_forD = Variable(label_forD.long())

            # Domain labels
            domain_labels_src_real = torch.LongTensor(Dout_sz[0], Dout_sz[2], Dout_sz[3]).zero_()
            domain_labels_src_fake = torch.LongTensor(Dout_sz[0], Dout_sz[2], Dout_sz[3]).zero_() + 1
            domain_labels_tgt_real = torch.LongTensor(Dout_sz[0], Dout_sz[2], Dout_sz[3]).zero_() + 2
            domain_labels_tgt_fake = torch.LongTensor(Dout_sz[0], Dout_sz[2], Dout_sz[3]).zero_() + 3

            domain_labels_src_real = Variable(domain_labels_src_real.cuda())
            domain_labels_src_fake = Variable(domain_labels_src_fake.cuda())
            domain_labels_tgt_real = Variable(domain_labels_tgt_real.cuda())
            domain_labels_tgt_fake = Variable(domain_labels_tgt_fake.cuda())

            # Updates.
            # There are three sets of updates - (1) Discriminator, (2) Generator and (3) F network

            # (1) Discriminator updates
            lossD_src_real_s = cross_entropy2d(outD_src_real_s, domain_labels_src_real, size_average=self.size_average)
            lossD_src_fake_s = cross_entropy2d(outD_src_fake_s, domain_labels_src_fake, size_average=self.size_average)
            lossD_src_real_c = cross_entropy2d(outD_src_real_c, label_forD,
                                               size_average=self.size_average)  # TODO,buggy
            lossD_tgt_real = cross_entropy2d(outD_tgt_real_s, domain_labels_tgt_real, size_average=self.size_average)
            lossD_tgt_fake = cross_entropy2d(outD_tgt_fake_s, domain_labels_tgt_fake, size_average=self.size_average)

            self.optimD.zero_grad()
            lossD = lossD_src_real_s + lossD_src_fake_s + lossD_src_real_c + lossD_tgt_real + lossD_tgt_fake
            lossD /= len(data_source)
            lossD.backward(retain_graph=True)
            self.optimD.step()

            # (2) Generator updates
            self.optimG.zero_grad()
            lossG_src_adv_s = cross_entropy2d(outD_src_fake_s, domain_labels_src_real, size_average=self.size_average)
            lossG_src_adv_c = cross_entropy2d(outD_src_fake_c, label_forD, size_average=self.size_average)
            lossG_tgt_adv_s = cross_entropy2d(outD_tgt_fake_s, domain_labels_tgt_real, size_average=self.size_average)
            lossG_src_mse = F.l1_loss(outG_src, data_source_forD)
            lossG_tgt_mse = F.l1_loss(outG_tgt, data_target_forD)

            lossG = lossG_src_adv_c + 0.1 * (lossG_src_adv_s + lossG_tgt_adv_s) + self.l1_weight * (
            lossG_src_mse + lossG_tgt_mse)
            lossG /= len(data_source)
            lossG.backward(retain_graph=True)
            self.optimG.step()

            # (3) F network updates
            self.optim.zero_grad()
            lossC = cross_entropy2d(score, labels_source, size_average=self.size_average)
            lossF_src_adv_s = cross_entropy2d(outD_src_fake_s, domain_labels_tgt_real, size_average=self.size_average)
            lossF_tgt_adv_s = cross_entropy2d(outD_tgt_fake_s, domain_labels_src_real, size_average=self.size_average)
            lossF_src_adv_c = cross_entropy2d(outD_src_fake_c, label_forD, size_average=self.size_average)

            lossF = lossC + self.adv_weight * (lossF_src_adv_s + lossF_tgt_adv_s) + self.c_weight * lossF_src_adv_c
            lossF /= len(data_source)
            lossF.backward()
            self.optim.step()

            if np.isnan(float(lossD.data[0])):
                raise ValueError('lossD is nan while training')
            if np.isnan(float(lossG.data[0])):
                raise ValueError('lossG is nan while training')
            if np.isnan(float(lossF.data[0])):
                raise ValueError('lossF is nan while training')

            # Computing metrics for logging
            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = labels_source.data.cpu().numpy()
            for lt, lp in zip(lbl_true, lbl_pred):
                acc, acc_cls, mean_iu, fwavacc = \
                    torchfcn.utils.label_accuracy_score(
                        [lt], [lp], n_class=self.n_class)
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            # Logging
            if self.iteration % 100 == 0:
                logger.info(
                    "epoch: {}/{}, iteration:{}, lossF:{}, mIoU :{}".format(self.epoch, self.max_epoch, self.iteration,
                                                                            lossF.data[0], metrics[2]))
            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [lossF.data[0]] + \
                      metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

            # Validating periodically
            if self.iteration % self.interval_validate == 0 and self.iteration > 0:
                out_recon = osp.join(self.out, 'visualization_viz')
                if not osp.exists(out_recon):
                    os.makedirs(out_recon)
                generations = []

                # Saving generated source and target images
                source_img = self.val_loader.dataset.untransform(data_source.data.cpu().numpy().squeeze())
                target_img = self.val_loader.dataset.untransform(data_target.data.cpu().numpy().squeeze())
                outG_src_ = (outG_src) * 255.0
                outG_tgt_ = (outG_tgt) * 255.0
                outG_src_ = outG_src_.data.cpu().numpy().squeeze().transpose((1, 2, 0))[:, :, ::-1].astype(np.uint8)
                outG_tgt_ = outG_tgt_.data.cpu().numpy().squeeze().transpose((1, 2, 0))[:, :, ::-1].astype(np.uint8)

                generations.append(source_img)
                generations.append(outG_src_)
                generations.append(target_img)
                generations.append(outG_tgt_)
                out_file = osp.join(out_recon, 'iter%012d_src_target_recon.png' % self.iteration)
                scipy.misc.imsave(out_file, fcn.utils.get_tile_image(generations))

                # Validation
                self.validate()
                self.model.train()
                self.netG.train()

    def train(self):
        """
        Function to train our model. Calls train_epoch function every epoch.
        Also performs learning rate annhealing
        """
        max_epoch = int(math.ceil(self.max_iter / min(len(self.train_loader), len(self.target_loader))))
        self.max_epoch = max_epoch
        logger.info("max epoch: {}".format(max_epoch))
        for epoch in tqdm.tqdm(range(self.epoch, max_epoch), desc='Train {}/{}'.format(self.epoch, max_epoch)):
            self.epoch = epoch
            if self.epoch % 8 == 0 and self.epoch > 0:
                logger.info("change learning rate!!!")
                self.optim = step_scheduler(self.optim, self.epoch)

            self.train_epoch()
            if self.iteration >= self.max_iter:
                break

    

if __name__ == '__main__':
    main()
