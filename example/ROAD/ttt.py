# Author: Tao Hu <taohu620@gmail.com>

import torch
from torch.autograd import Variable
x = Variable(torch.ones(2, 2), requires_grad = True)
y = x ** 2
y.backward(torch.ones(2, 2), retain_variables=True)
print "first backward of x is:"
print x.grad
y.backward(2*torch.ones(2, 2), retain_variables=True)
print "second backward of x is:"
print x.grad
