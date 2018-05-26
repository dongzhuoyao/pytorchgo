# Acknowledgement: most codes are borrowed from: https://github.com/isht7/pytorch-deeplab-resnet

import torch.nn as nn
import torch
import numpy as np
import fcn, os
from pytorchgo.utils import logger
affine_par = True


def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
	for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
	    padding = 2
        elif dilation_ == 4:
	    padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self,dilation_series,padding_series,NoLabels):
        super(Classifier_Module, self).__init__()
	self.conv2d_list = nn.ModuleList()
	for dilation,padding in zip(dilation_series,padding_series):
	    self.conv2d_list.append(nn.Conv2d(2048,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)


    def forward(self, x):
	out = self.conv2d_list[0](x)
	for i in range(len(self.conv2d_list)-1):
	    out += self.conv2d_list[i+1](x)
        return out



class ResNet(nn.Module):
    def __init__(self, block, layers,NoLabels):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__ = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 4)
	self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],NoLabels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation_=dilation__))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,NoLabels):
	return block(dilation_series,padding_series,NoLabels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
	x = self.layer5(x)

        return x

class MS_Deeplab(nn.Module):
    pretrained_model = \
        os.path.expanduser('~/data/models/pytorch/MS_DeepLab_resnet_pretrained_COCO_init.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='https://dongzhuoyao.oss-cn-qingdao.aliyuncs.com/MS_DeepLab_resnet_pretrained_COCO_init.pth',
            path=cls.pretrained_model,
            md5='a5720af006c01fd69b3325da36e1c9ed',
        )


    def __init__(self,block,NoLabels):
	super(MS_Deeplab,self).__init__()
	self.Scale = ResNet(block,[3, 4, 23, 3],NoLabels)   #changed to fix #4

    def forward(self,x):
        input_size = x.size()[2]
        self.interp75 = nn.UpsamplingBilinear2d(size = (int(input_size * 0.75) + 1, int(input_size * 0.75) + 1))
        self.interp50 = nn.UpsamplingBilinear2d(size = (int(input_size * 0.5) + 1, int(input_size * 0.5) + 1))
        self.interp3 = nn.UpsamplingBilinear2d(size = (  outS(input_size),   outS(input_size)   ))

        self.interp_origin = nn.UpsamplingBilinear2d(size = (x.size()[2],x.size()[3]))
        out = []
        x75 = self.interp75(x)
        x50 = self.interp50(x)
        out.append(self.Scale(x))	# for original scale
        out.append(self.interp3(self.Scale(x75)))	# for 0.75x scale
        out.append(self.Scale(x50))	# for 0.5x scale
        x2Out_interp = out[1]
        x3Out_interp = self.interp3(out[2])
        temp1 = torch.max(out[0],x2Out_interp)
        out.append(torch.max(temp1,x3Out_interp))
        #return out # here, for simplicity, we only use first output, which is original output size
        return self.interp_origin(out[-1])

    def optimizer_params(self, base_lr):# for optimizer usage
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': base_lr},
         {'params': self.get_10x_lr_params(), 'lr': 10 * base_lr}]

    def get_1x_lr_params_NOscale(model):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(model.Scale.conv1)
        b.append(model.Scale.bn1)
        b.append(model.Scale.layer1)
        b.append(model.Scale.layer2)
        b.append(model.Scale.layer3)
        b.append(model.Scale.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(model):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """

        b = []
        b.append(model.Scale.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i


def Res_Deeplab(NoLabels=21, pretrained = False):
    model = MS_Deeplab(Bottleneck,NoLabels)
    if pretrained:
        logger.info("initializing pretrained deeplabv2 model....")
        model_file = MS_Deeplab.download()
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':
    i = 120
    print(outS(i))
