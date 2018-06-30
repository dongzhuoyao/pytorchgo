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