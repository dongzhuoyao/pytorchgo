## Few Shot Det(**python3.5,  pytorch  0.3.x**)





max 25k iterations, or it will overfit!
```
Traceback (most recent call last):
  File "train.baseline.5e-4.channel6.fold2.size512.py", line 365, in <module>
    train()
  File "train.baseline.5e-4.channel6.fold2.size512.py", line 202, in train
    loss_l, loss_c = criterion(out, targets)
  File "/home/hutao/.virtualenvs/pytorch03-py35/lib/python3.5/site-packages/torch/nn/modules/module.py", line 357, in __call__
    result = self.forward(*input, **kwargs)
  File "/data4/hutao/lab/pytorchgo/example/few-shot-det/code/layers/modules/multibox_loss.py", line 92, in forward
    loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))#????
  File "/data4/hutao/lab/pytorchgo/example/few-shot-det/code/layers/box_utils.py", line 165, in log_sum_exp
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max
RuntimeError: value cannot be converted to type float without overflow: inf
```
|arch|mAP|
|----|----|
train.baseline.5e-4.channel6|23.48|
train.baseline.5e-4.channel6.fold1|30.87|
train.baseline.5e-4.channel6.fold2|18.92|
train.baseline.5e-4.channel6.fold3|21.32|
train.baseline.5e-4.channel6.size512|15.5,**U shape result**|
train.baseline.5e-4.channel6.size512.refined_lr|19.09|
train.baseline.5e-4.channel6.size512.refined_lr.fuse_conv5|17.6|
train.baseline.5e-4.channel6.size512.refined_lr.fuse_conv45|17.82|
train.baseline.5e-4.channel6.size512.refined_lr.fuse_conv345|18.28|
train.baseline.5e-4.channel6.size512.refined_lr.aug|18.95|
train.baseline.5e-4.channel6.size512.refined_lr.aug.40k|23.02|
|----|----|
train.baseline.5e-4.channel6.fold1.size512|27.22|
train.baseline.5e-4.channel6.fold1.size512.refined_lr|30.84|
train.baseline.5e-4.channel6.fold1.size512.refined_lr.aug|33.38|
train.baseline.5e-4.channel6.fold1.size512.refined_lr.fuse_conv4|29.09|
train.baseline.5e-4.channel6.fold1.size512.refined_lr.fuse_conv5|25.35|
train.baseline.5e-4.channel6.fold1.size512.refined_lr.fuse_conv345|28.13|
|----|----|
train.baseline.5e-4.channel6.fold2.size512|14.03|
train.baseline.5e-4.channel6.fold2.size512.refined_lr|18.9|
train.baseline.5e-4.channel6.fold2.size512.refined_lr.fuse_conv4|16.83|
train.baseline.5e-4.channel6.fold2.size512.refined_lr.fuse_conv345|22.83|
train.baseline.5e-4.channel6.fold2.size512.refined_lr.aug|22.16|
train.baseline.5e-4.channel6.fold2.size512.refined_lr.aug.40k|19.13|
|----|----|
train.baseline.5e-4.channel6.fold3.size512|18.01|
train.baseline.5e-4.channel6.fold3.size512.refined_lr|22.3|
train.baseline.5e-4.channel6.fold3.size512.refined_lr.aug|25.35|
train.baseline.5e-4.channel6.fold3.size512.refined_lr.fuse_conv4|22.48|
train.baseline.5e-4.channel6.fold3.size512.refined_lr.fuse_conv5|21.74|
train.baseline.5e-4.channel6.fold3.size512.refined_lr.fuse_conv345|24.07|





python2.7 error:
```
File "train.cs_car.512.py", line 172, in train
    loss.backward()
  File "/home/hutao/.virtualenvs/pytorch03/local/lib/python2.7/site-packages/torch/autograd/variable.py", line 167, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, retain_variables)
  File "/home/hutao/.virtualenvs/pytorch03/local/lib/python2.7/site-packages/torch/autograd/__init__.py", line 99, in backward
    variables, grad_variables, retain_graph)
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
```