import argparse
import torch

from torchfcn.trainer_ROAD_distill_full import MyTrainer_ROAD
from pytorchgo.utils.pytorch_utils import model_summary, optimizer_summary
from pytorchgo.utils.weight_init import weights_init


from torch.utils import data, model_zoo
import math
import os
import os.path as osp
import shutil
import numpy as np
from torch.autograd import Variable
import tqdm
import itertools
import torchfcn
from util_fns import get_parameters
from pytorchgo.loss.loss import CrossEntropyLoss2dWithLogits_Seg, Diff2d
from pytorchgo.utils.pytorch_utils import step_scheduler
from pytorchgo.utils import logger


class_num = 19
image_size=[641,641]#[768,384] #[641,641]#[640, 320]
image_size_target = [641,641]#[1024,512]

max_epoch = 30
base_lr = 1.0e-4
dis_lr = 1e-3
base_lr_schedule = [(20,1e-5),(25,1e-6)]
dis_lr_schedule = [(20,1e-4),(25,1e-5)]

Deeplabv2_restore_from = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'

def main():
    logger.auto_set_dir()
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/home/hutao/lab/pytorchgo/example/ROAD/data', help='Path to source dataset')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--max_epoch', type=int, default=max_epoch, help='Number of training iterations')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use | SGD, Adam')
    parser.add_argument('--lr', type=float, default=base_lr, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.99, help='Momentum for SGD')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--model', type=str, default='vgg16')
    parser.add_argument('--iter_size', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=1)

    args = parser.parse_args()
    print(args)


    gpu = args.gpu


    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        logger.info("random seed 1337")
        torch.cuda.manual_seed(1337)

    # Defining data loaders
    

    kwargs = {'num_workers': 4, 'pin_memory': True,'drop_last':True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SYNTHIA('SYNTHIA', args.dataroot, split='train', transform=True, image_size=image_size),
        batch_size=args.batchSize, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.CityScapes('cityscapes', args.dataroot, split='val', transform=True, image_size=image_size_target),
        batch_size=1, shuffle=False)

    target_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.CityScapes('cityscapes', args.dataroot, split='train', transform=True, image_size=image_size_target),
        batch_size=args.batchSize, shuffle=True)


    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if args.model == "vgg16":
        model = origin_model =  torchfcn.models.Seg_model(n_class=class_num)
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    elif args.model == "deeplabv2":#TODO may have problem!
        model =  origin_model = torchfcn.models.Res_Deeplab(num_classes=class_num)
        saved_state_dict = model_zoo.load_url(Deeplabv2_restore_from)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not class_num == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)
    else:
        raise ValueError("only support vgg16, deeplabv2!")


    model_fix = torchfcn.models.Seg_model(n_class=class_num)
    for param in model_fix.parameters():
        param.requires_grad = False



    netD = torchfcn.models.Domain_classifer(reverse=False)
    netD.apply(weights_init)

    model_summary(model)
    model_summary(netD)



    if cuda:
        model = model.cuda()
        netD = netD.cuda()
        
    # Defining optimizer
    
    if args.optimizer == 'SGD':
        raise ValueError("SGD is not prepared well..")
        optim = torch.optim.SGD(
            [
                {'params': get_parameters(model, bias=False)},
                {'params': get_parameters(model, bias=True),
                 'lr': args.lr * 2,
                 'weight_decay': args.weight_decay},
            ],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        if args.model == "vgg16":
            optim = torch.optim.Adam(
                [
                    {'params': get_parameters(model, bias=False),
                     'weight_decay': args.weight_decay},
                    {'params': get_parameters(model, bias=True),
                     'lr': args.lr * 2,
                     'weight_decay': args.weight_decay},
                ],
                lr=args.lr,
                betas=(args.beta1, 0.999))
        elif args.model == "deeplabv2":
            optim = torch.optim.Adam(
                origin_model.optim_parameters(args.lr),
                lr=args.lr,
                betas=(args.beta1, 0.999),
                weight_decay = args.weight_decay)
        else:
            raise
    else:
        raise ValueError('Invalid optmizer argument. Has to be SGD or Adam')


    optimD = torch.optim.Adam(netD.parameters(), lr=dis_lr, weight_decay = args.weight_decay, betas=(0.7, 0.999))

    optimizer_summary([optim,optimD])



    trainer = MyTrainer_ROAD(
        cuda=cuda,
        model=model,
        model_fix = model_fix,
        netD=netD,
        optimizer=optim,
        optimizerD=optimD,
        train_loader=train_loader,
        target_loader=target_loader,
        val_loader=val_loader,
        batch_size=args.batchSize,
        image_size=image_size
    )
    trainer.epoch = 0
    trainer.iteration = 0
    trainer.train()





class MyTrainer_ROAD(object):
    def __init__(self, cuda, model, model_fix, netD, optimizer, optimizerD,
                 train_loader, target_loader, val_loader,
                  image_size, batch_size,
                 size_average=True, loss_print_interval = 500):
        self.cuda = cuda
        self.model = model
        self.model_fix = model_fix
        self.netD = netD
        self.optim = optimizer
        self.optimD = optimizerD
        self.batch_size = batch_size
        self.iter_size = 1

        self.loss_print_interval =loss_print_interval
        self.train_loader = train_loader
        self.target_loader = target_loader
        self.val_loader = val_loader

        self.image_size = tuple(image_size)
        self.n_class = len(self.train_loader.dataset.class_names)

        self.size_average = size_average


        self.epoch = 0
        self.iteration = 0
        self.best_mean_iu = 0

    def validate(self):
        """
        Function to validate a training model on the val split.
        """
        logger.info("start validation....")
        val_loss = 0
        label_trues, label_preds = [], []

        # Evaluation
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Validation iteration = {},epoch={}'.format(self.iteration,self.epoch),
                leave=False):

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            score = self.model(data)

            loss = CrossEntropyLoss2dWithLogits_Seg(score, target, size_average=self.size_average)

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

        logger.info("iteration={},epoch={},validation mIoU = {}".format(self.iteration, self.epoch, mean_iu))

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
        def set_requires_grad(seg, dis):
            for param in self.model.parameters():
                param.requires_grad = seg

            for param in self.netD.parameters():
                param.requires_grad = dis

        self.train_loader_iter = enumerate(self.train_loader)
        self.target_loader_iter = enumerate(self.target_loader)

        for batch_idx in tqdm.tqdm(
                range(self.iters_per_epoch),
                total=self.iters_per_epoch,
                desc='Train epoch = {}/{}'.format(self.epoch, self.max_epoch)):
            self.iteration = batch_idx + self.epoch * self.iters_per_epoch

            self.optim.zero_grad()
            self.optimD.zero_grad()

            sum_l_seg = 0
            sum_distill_loss =0
            sum_src_dis_loss = 0
            sum_target_dis_loss = 0

            #for sub_iter in range(self.iter_size):

            _, source_batch = self.train_loader_iter.next()
            source_data, source_labels = source_batch

            _, target_batch = self.target_loader_iter.next()
            target_data, _ = target_batch


            if self.cuda:
                source_data, source_labels = source_data.cuda(), source_labels.cuda()
                target_data = target_data.cuda()

            source_data, source_labels = Variable(source_data), Variable(source_labels)
            target_data = Variable(target_data)

            #######################train G
            set_requires_grad(seg=True, dis=False)
            # Source domain
            score = self.model(source_data)
            l_seg = CrossEntropyLoss2dWithLogits_Seg(score, source_labels, class_num=class_num, size_average=self.size_average)

            sum_l_seg += l_seg.data[0]

            # target domain
            seg_target_score = self.model(target_data)
            modelfix_target_score = self.model_fix(target_data)

            diff2d = Diff2d()
            distill_loss = diff2d(seg_target_score, modelfix_target_score)
            sum_distill_loss += distill_loss.data[0]

            seg_loss = l_seg + 10 * distill_loss

            seg_loss.backward()





            #########################train D
            set_requires_grad(seg=False, dis=True)
            src_discriminate_result = self.netD(score.detach())
            target_discriminate_result = self.netD(seg_target_score.detach())

            bce_loss = torch.nn.BCEWithLogitsLoss()

            src_dis_loss = bce_loss(src_discriminate_result,
                                           Variable(torch.FloatTensor(src_discriminate_result.data.size()).fill_(1)).cuda())

            target_dis_loss = bce_loss(target_discriminate_result,
                                              Variable(torch.FloatTensor(target_discriminate_result.data.size()).fill_(0)).cuda())

            sum_src_dis_loss += src_dis_loss.data[0]
            sum_target_dis_loss += target_dis_loss.data[0]

            dis_loss = src_dis_loss + target_dis_loss# this loss has been inversed!!

            dis_loss.backward()

            if np.isnan(float(dis_loss.data[0])):
                raise ValueError('dis_loss is nan while training')
            if np.isnan(float(seg_loss.data[0])):
                raise ValueError('total_loss is nan while training')



            self.optim.step()
            self.optimD.step()


            if self.iteration % self.loss_print_interval == 0:
                logger.info("L_SEG={}, Distill_LOSS={}, src dis loss={}, target dis loss={}".format(sum_l_seg, sum_distill_loss,
                                                                               sum_src_dis_loss,sum_target_dis_loss))



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
        self.max_epoch = args.max_epoch
        for epoch in tqdm.trange(self.epoch, args.max_epoch, desc='Train'):
            self.epoch = epoch
            self.optim = step_scheduler(self.optim, self.epoch, base_lr_schedule, "base model")
            self.optimD = step_scheduler(self.optimD, self.epoch,  dis_lr_schedule, "discriminater model")

            self.model.train()
            self.netD.train()
            self.train_epoch()

            self.model.eval()
            self.validate()
            self.model.train()  # return to training mode


if __name__ == '__main__':
    main()
