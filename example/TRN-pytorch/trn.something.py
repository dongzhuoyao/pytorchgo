import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
from pytorchgo.utils import logger
from tqdm import tqdm

try:
    from .dataset import TSNDataSet
    from .models import TSN
    from .transforms import *
    from . import datasets_video


except Exception:
    from dataset import TSNDataSet
    from models import TSN
    from transforms import *
    import datasets_video


best_prec1 = 0

def main():
    logger.auto_set_dir()

    global args, best_prec1

    import argparse
    parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
    parser.add_argument('--dataset', type=str,default="something", choices=['something', 'jester', 'moments'])
    parser.add_argument('--modality', type=str, default="RGB", choices=['RGB', 'Flow'])
    parser.add_argument('--train_list', type=str, default="")
    parser.add_argument('--val_list', type=str, default="")
    parser.add_argument('--root_path', type=str, default="")
    parser.add_argument('--store_name', type=str, default="")
    # ========================= Model Configs ==========================
    parser.add_argument('--arch', type=str, default="BNInception")
    parser.add_argument('--num_segments', type=int, default=3)
    parser.add_argument('--consensus_type', type=str, default='avg')
    parser.add_argument('--k', type=int, default=3)

    parser.add_argument('--dropout', '--do', default=0.8, type=float,
                        metavar='DO', help='dropout ratio (default: 0.5)')
    parser.add_argument('--loss_type', type=str, default="nll",
                        choices=['nll'])
    parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")

    # ========================= Learning Configs ==========================
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                        metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--clip_gradient', '--gd', default=20, type=float,
                        metavar='W', help='gradient norm clipping (default: disabled)')
    parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

    # ========================= Monitor Configs ==========================
    parser.add_argument('--print_freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--eval_freq', '-ef', default=5, type=int,
                        metavar='N', help='evaluation frequency (default: 5)')

    # ========================= Runtime Configs ==========================
    parser.add_argument('-j', '--workers', default=30, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--snapshot_pref', type=str, default="")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpu', type=str, default='4')
    parser.add_argument('--flow_prefix', default="", type=str)
    parser.add_argument('--root_log', type=str, default='log')
    parser.add_argument('--root_model', type=str, default='model')
    parser.add_argument('--root_output', type=str, default='output')

    args = parser.parse_args()

    args.num_segments = 7
    args.batch_size = 64
    args.consensus_type = "TRN"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device_ids = [int(id) for id in args.gpu.split(',')]
    assert len(device_ids) >1, "TRN must run with GPU_num > 1"

    args.root_log = logger.get_logger_dir()
    args.root_model = logger.get_logger_dir()
    args.root_output = logger.get_logger_dir()

    categories, args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.modality)
    num_class = len(categories)


    args.store_name = '_'.join(['TRN', args.dataset, args.modality, args.arch, args.consensus_type, 'segment%d'% args.num_segments])
    logger.info('storing name: ' + args.store_name)

    model = TSN(num_class=num_class, num_segments=args.num_segments, modality=args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)#TODO, , device_ids=[int(id) for id in args.gpu.split(',')]

    if torch.cuda.is_available():
       model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            logger.info(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        logger.info('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult']))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    from pytorchgo.utils.pytorch_utils import model_summary,optimizer_summary
    model_summary(model)
    optimizer_summary(optimizer)


    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)#only data_parellel exists model.module object!!!
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader),total=len(train_loader), desc="train epoch={}/{}".format(epoch, args.epochs)):

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                logger.warn("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()


        if i % args.print_freq == 0:
            logger.info('iter=[{}/{}], lr={:0.5f}  Loss={:0.4f} Prec@1={:0.3f} Prec@5={:0.3f}'.format(
                        i, len(train_loader), optimizer.param_groups[-1]['lr'], losses.avg, top1.avg, top5.avg))




def validate(val_loader, model, criterion, iter):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Test: [{}/{}] Time {:0.3f} Loss {:0.4f} Prec@1={:0.3f} Prec@5={:0.3f}'.format(
                   i, len(val_loader), batch_time, losses.avg, top1.avg, top5.avg))



    logger.info('Testing Results: Prec@1={:0.3f} Prec@5={:0.3f} Loss={:0.5f} Best Prec@1={:0.3f}'.format(top1.avg, top5.avg, losses.avg, best_prec1))



    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name),'%s/%s_best.pth.tar' % (args.root_model, args.store_name))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



if __name__ == '__main__':
    main()
