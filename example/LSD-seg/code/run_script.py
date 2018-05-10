import os
root = '/home/hutao/lab/pytorchgo/example/LSD-seg/data'
cmd = 'python train.py --dataroot ' + root + ' --gpu 2 --method LSD'
os.system(cmd)
