import torch
import torch.nn as nn
import torch.nn.functional as F



def MSE_Loss(inputs1, inputs2):
        return torch.mean((inputs1 - inputs2) ** 2)

# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


def CrossEntropyLoss2d_Seg(input, target, class_num, weight=None, size_average=True):
    """
    Function to compute pixelwise cross-entropy for 2D image. This is the segmentation loss.
    Args:
        input: input tensor of shape (minibatch x num_channels x h x w)
        target: 2D label map of shape (minibatch x h x w)
        weight (optional): tensor of size 'C' specifying the weights to be given to each class
        size_average (optional): boolean value indicating whether the NLL loss has to be normalized
            by the number of pixels in the image
    """

    # input: (n, c, h, w),
    # target: (n, h, w),1x40x80
    n, c, h, w = input.size()

    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)


    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)# log_p: (n*h*w, c)
    try:
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) < class_num]
    except:
        import traceback
        traceback.print_exc()
        print("Exception: {}".format(target.size()))
    log_p = log_p.view(-1, c)

    # target: (n*h*w,)
    mask = target < class_num
    target = target[mask]
    target = torch.squeeze(target)
    try:
        loss = F.nll_loss(log_p, target, weight=weight, size_average=size_average)
    except:
        print("log_p size: {}".format(log_p.size()))
        print("target size: {}".format(target.size()))
        import traceback
        traceback.print_exc()
        import ipdb
        ipdb.set_trace()
        # log_p: 1x19x40x80
        # target: Variable containing:[torch.cuda.LongTensor with no dimension]
    # if size_average:
    #    loss /= mask.data.sum()

    return loss


class BalanceLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BalanceLoss2d, self).__init__()
        self.weight = weight

    def forward(self, inputs1, inputs2, class_num):
        prob1 = F.softmax(inputs1)[0, :class_num]
        prob2 = F.softmax(inputs2)[0, :class_num]
        print(prob1)
        prob1 = torch.mean(prob1, 0)
        prob2 = torch.mean(prob2, 0)
        print(prob1)
        entropy_loss = - torch.mean(torch.log(prob1 + 1e-6))
        entropy_loss -= torch.mean(torch.log(prob2 + 1e-6))
        return(entropy_loss)


class Entropy(nn.Module):
    def __init__(self, weight=None):
        super(Entropy, self).__init__()
        self.weight = weight

    def forward(self, inputs1, class_num):
        prob1 = F.softmax(inputs1[0, :class_num])
        entropy_loss = torch.mean(torch.log(prob1))  # torch.mean(torch.mean(torch.log(prob1),1),0
        return entropy_loss

class Diff2d(nn.Module):
    def __init__(self, weight=None):
        super(Diff2d, self).__init__()
        self.weight = weight

    def forward(self, inputs1, inputs2):
        return torch.mean(torch.abs(F.softmax(inputs1) - F.softmax(inputs2)))




class Symkl2d(nn.Module):
    def __init__(self, weight=None, n_target_ch=21, size_average=True):
        super(Symkl2d, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.n_target_ch = 20
    def forward(self, inputs1, inputs2):
        self.prob1 = F.softmax(inputs1)
        self.prob2 = F.softmax(inputs2)
        self.log_prob1 = F.log_softmax(self.prob1)
        self.log_prob2 = F.log_softmax(self.prob2)

        loss = 0.5 * (F.kl_div(self.log_prob1, self.prob2, size_average=self.size_average)
                      + F.kl_div(self.log_prob2, self.prob1, size_average=self.size_average))

        return loss







# this may be unstable sometimes.Notice set the size_average
def CrossEntropy2d(input, target, weight=None, size_average=False):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss


def get_prob_distance_criterion(criterion_name, n_class=None):
    if criterion_name == 'diff':
        criterion = Diff2d()
    elif criterion_name == "symkl":
        criterion = Symkl2d(n_target_ch=n_class)
    elif criterion_name == "nmlsymkl":
        criterion = Symkl2d(n_target_ch=n_class, size_average=True)
    else:
        raise NotImplementedError()

    return criterion
