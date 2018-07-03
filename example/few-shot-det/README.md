## Few Shot Det(**python3.5,  pytorch  0.3.x**)




##

|arch|mAP|
|----|----|
|lre-3|unstable, terminated|
|train.baseline|5.64|
|train.baseline.channel4|4.31|
|train.baseline.channel5|5.36|
|train.baseline.channel6|5.93|
|train.baseline.5e-4,intermediate result(top_k=200)|6.96|
|train.baseline.5e-4,intermediate result(top_k=200,with score)|9.61|
|train.baseline.5e-4,intermediate result(top_k=20,with score)|9.63|
train.baseline.fold1|8.08|
train.baseline.fold2|4.64|



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
train.baseline.5e-4.channel6.fold1.size512|27.22|
train.baseline.5e-4.channel6.fold2.size512|14.03|



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