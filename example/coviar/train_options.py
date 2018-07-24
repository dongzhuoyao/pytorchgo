"""Training options."""

import argparse

parser = argparse.ArgumentParser(description="CoViAR")

# Data.
parser.add_argument('--data-name', type=str, default="hmdb51", choices=['ucf101', 'hmdb51'],
                    help='dataset name.')
parser.add_argument('--data-root', type=str, default='/data4/hutao/dataset/hmdb51_videos_mpeg4',
                    help='root of data directory.')
parser.add_argument('--train-list', type=str, default='data/datalists/hmdb51_split1_train.txt',
                    help='training example list.')
parser.add_argument('--test-list', type=str, default='data/datalists/hmdb51_split1_test.txt',
                    help='testing example list.')

# Model.
parser.add_argument('--representation', type=str, default="mv", choices=['iframe', 'mv', 'residual'],
                    help='data representation.')
parser.add_argument('--arch', type=str, default="resnet18",
                    help='base architecture.')
parser.add_argument('--num_segments', type=int, default=3,
                    help='number of TSN segments.')
parser.add_argument('--no-accumulation', action='store_true',
                    help='disable accumulation of motion vectors and residuals.')

# Training.
parser.add_argument('--epochs', default=360, type=int,
                    help='number of training epochs.')
parser.add_argument('--batch-size', default=80, type=int,
                    help='batch size.')
parser.add_argument('--lr', default=0.005, type=float,
                    help='base learning rate.')
parser.add_argument('--lr-steps', default=[120,200,280], type=float, nargs="+",
                    help='epochs to decay learning rate.')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    help='lr decay factor.')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay.')

# Log.
parser.add_argument('--eval-freq', default=5, type=int,
                    help='evaluation frequency (epochs).')
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loader workers.')
parser.add_argument('--model-prefix', type=str, default="hmdb51_mv_model",
                    help="prefix of model name.")
parser.add_argument('--gpus', type=str, default='7',
                    help='gpu ids.')
