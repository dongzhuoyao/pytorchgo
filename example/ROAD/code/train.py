import argparse
import os
import os.path as osp
import sys
import torch
import torchfcn
from util_fns import get_log_dir
from util_fns import get_parameters
from util_fns import weights_init
from pytorchgo.utils import logger
from torchfcn.trainer_ROAD import MyTrainer_ROAD
class_num = 19


def main():
    logger.auto_set_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/home/hutao/lab/pytorchgo/example/ROAD/data', help='Path to source dataset')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--num_iters', type=int, default=100000, help='Number of training iterations')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use | SGD, Adam')
    parser.add_argument('--lr', type=float, default=1.0e-5, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.99, help='Momentum for SGD')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--interval_validate', type=int, default=500, help='Period for validation. Model is validated every interval_validate iterations')
    parser.add_argument('--gpu', type=int, default=0)
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
    
    image_size=[321,321]#[640, 320]
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SYNTHIA('SYNTHIA', args.dataroot, split='train', transform=True, image_size=image_size),
        batch_size=args.batchSize, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SYNTHIA('SYNTHIA', args.dataroot, split='val', transform=True, image_size=image_size),
        batch_size=args.batchSize, shuffle=False, **kwargs)
    target_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.CityScapes('cityscapes', args.dataroot, split='train', transform=True, image_size=image_size),
        batch_size=args.batchSize, shuffle=True)

    # Defining models

    start_epoch = 0
    start_iteration = 0

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    model = torchfcn.models.Seg_model(n_class=class_num)
    model_fix = torchfcn.models.Seg_model(n_class=class_num) #TODO fix weight
    for param in model_fix.parameters():
        param.requires_grad = False

    netD = torchfcn.models.Domain_classifer()
    netD.apply(weights_init)


    vgg16 = torchfcn.models.VGG16(pretrained=True)
    model.copy_params_from_vgg16(vgg16)

    if cuda:
        model = model.cuda()
        netD = netD.cuda()
        
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
    

    optimD = torch.optim.Adam(netD.parameters(), lr=0.0001, betas=(0.7, 0.999))


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
        max_iter=args.num_iters,
        interval_validate=args.interval_validate,
        image_size=image_size
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()
    

if __name__ == '__main__':
    main()
