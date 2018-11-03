import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
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

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
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

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class Residual_Covolution(nn.Module):
    def __init__(self, icol, ocol, num_classes):
        super(Residual_Covolution, self).__init__()
        self.conv1 = nn.Conv2d(icol, ocol, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv2 = nn.Conv2d(ocol, num_classes, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv3 = nn.Conv2d(num_classes, ocol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv4 = nn.Conv2d(ocol, icol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dow1 = self.conv1(x)
        dow1 = self.relu(dow1)
        seg = self.conv2(dow1)
        inc1 = self.conv3(seg)
        add1 = dow1 + self.relu(inc1)
        inc2 = self.conv4(add1)
        out = x + self.relu(inc2)
        return out, seg

class Residual_Refinement_Module(nn.Module):

    def __init__(self, num_classes):
        super(Residual_Refinement_Module, self).__init__()
        self.RC1 = Residual_Covolution(2048, 512, num_classes)
        self.RC2 = Residual_Covolution(2048, 512, num_classes)

    def forward(self, x):
        x, seg1 = self.RC1(x)
        _, seg2 = self.RC2(x)
        return [seg1, seg1+seg2]

class ResNet_Refine(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet_Refine, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = Residual_Refinement_Module(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

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

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

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
    def __init__(self,block,num_classes):
        super(MS_Deeplab,self).__init__()
        self.Scale = ResNet(block,[3, 4, 23, 3],num_classes)   #changed to fix #4 

    def forward(self,x):
        output = self.Scale(x) # for original scale
        output_size = output.size()[2]
        input_size = x.size()[2]

        self.interp1 = nn.Upsample(size=(int(input_size*0.75)+1, int(input_size*0.75)+1), mode='bilinear')
        self.interp2 = nn.Upsample(size=(int(input_size*0.5)+1, int(input_size*0.5)+1), mode='bilinear')
        self.interp3 = nn.Upsample(size=(output_size, output_size), mode='bilinear')

        x75 = self.interp1(x)
        output75 = self.interp3(self.Scale(x75)) # for 0.75x scale

        x5 = self.interp2(x)
        output5 = self.interp3(self.Scale(x5))	# for 0.5x scale

        out_max = torch.max(torch.max(output, output75), output5)
        return [output, output75, output5, out_max]

def Res_Ms_Deeplab(num_classes=21):
    model = MS_Deeplab(Bottleneck, num_classes)
    return model

def Res_Deeplab(num_classes=21, is_refine=False):
    if is_refine:
        model = ResNet_Refine(Bottleneck,[3, 4, 6, 3], num_classes)
    else:
        model = ResNet(Bottleneck,[3, 4, 6, 3], num_classes)

    #load weight:
    #https://download.pytorch.org/models/resnet50-19c8e357.pth
    return model

def Res101_Deeplab(num_classes=21, is_refine=False):
    if is_refine:
        model = ResNet_Refine(Bottleneck,[3, 4, 23, 3], num_classes)
    else:
        model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)

    #load weight:
    #https://download.pytorch.org/models/resnet50-19c8e357.pth
    return model

class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(channel, channel,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(channel, affine = affine_par),
                nn.Sigmoid()
        )


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)#.view(b, c, 1, 1)

        return x * y
        #return x + x*y



class MyResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(MyResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layer1 = x = self.layer1(x)
        layer2 = x = self.layer2(x)
        layer3 = x = self.layer3(x)
        layer4 = x = self.layer4(x)
        x = self.layer5(x)

        return layer1,layer2,layer3,layer4,x


class HandInHandModel(nn.Module):

    def __init__(self, block, layers, teacher_class_num, student_class_num, annealing = False,get_anneal=None, netstyle=0):
        super(HandInHandModel, self).__init__()
        #teacher network
        self.teacher = MyResNet(block, layers, teacher_class_num)

        self.iter = 1.0
        self.annealing = annealing
        self.netstyle = netstyle


        self.get_anneal = get_anneal

        self.semodule1 = SELayer(channel=64*4)
        self.semodule2 = SELayer(channel=128*4)
        self.semodule3 = SELayer(channel=256*4)
        self.semodule4 = SELayer(channel=512*4)


        #student network
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],student_class_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)




    def forward(self, x):

        b1_layer1, b1_layer2, b1_layer3, b1_layer4, b1 = self.teacher(x)

        b2 = self.conv1(x)
        b2 = self.bn1(b2)
        b2 = self.relu(b2)
        b2 = self.maxpool(b2)

        if self.annealing:
            annealing_value = self.get_anneal(self.iter)
        else:
            annealing_value = 1

        if self.training:
            #print "update!"
            self.iter += 1
        #print "anneal: {}".format(annealing_value)
        if self.netstyle == 0:
            b2 = self.layer1(b2) + self.semodule1(b1_layer1)*annealing_value
            b2 = self.layer2(b2) + self.semodule2(b1_layer2)*annealing_value
            b2 = self.layer3(b2) + self.semodule3(b1_layer3)*annealing_value
            b2 = self.layer4(b2) + self.semodule4(b1_layer4)*annealing_value

        elif self.netstyle == 1:
            b2 = self.layer1(b2)
            b2 = self.layer2(b2)
            b2 = self.layer3(b2)
            b2 = self.layer4(b2) + self.semodule4(b1_layer4) * annealing_value
        elif self.netstyle == 2:
            b2 = self.layer1(b2)
            b2 = self.layer2(b2)
            b2 = self.layer3(b2)+ self.semodule3(b1_layer3)*annealing_value
            b2 = self.layer4(b2) + self.semodule4(b1_layer4) * annealing_value
        elif self.netstyle == 3:
            b2 = self.layer1(b2)
            b2 = self.layer2(b2) + self.semodule2(b1_layer2)*annealing_value
            b2 = self.layer3(b2)+ self.semodule3(b1_layer3)*annealing_value
            b2 = self.layer4(b2) + self.semodule4(b1_layer4) * annealing_value
        elif self.netstyle == 4:
            b2 = self.layer1(b2) + b1_layer1*annealing_value
            b2 = self.layer2(b2) + b1_layer2*annealing_value
            b2 = self.layer3(b2) + b1_layer3*annealing_value
            b2 = self.layer4(b2) + b1_layer4*annealing_value
        elif self.netstyle == 5:
            b2 = self.layer1(b2)
            b2 = self.layer2(b2) + b1_layer2*annealing_value
            b2 = self.layer3(b2) + b1_layer3*annealing_value
            b2 = self.layer4(b2) + b1_layer4*annealing_value
        elif self.netstyle == 6:
            b2 = self.layer1(b2)
            b2 = self.layer2(b2)
            b2 = self.layer3(b2) + b1_layer3*annealing_value
            b2 = self.layer4(b2) + b1_layer4*annealing_value
        elif self.netstyle == 7:
            b2 = self.layer1(b2)
            b2 = self.layer2(b2)
            b2 = self.layer3(b2)
            b2 = self.layer4(b2) + b1_layer4*annealing_value
        elif self.netstyle == 8:
            b2 = self.layer1(b2)
            b2 = self.layer2(b2)
            b2 = self.layer3(b2) + b1_layer3*annealing_value
            b2 = self.layer4(b2)
        else:
            raise

        b2 = self.layer5(b2)

        return b1, b2



class HandInHandModel_Hourglass(nn.Module):

    def __init__(self, block, layers, teacher_class_num, student_class_num, hourglass_depth=3, annealing = False,get_anneal=None, netstyle=0, compress_ratio=8):
        super(HandInHandModel_Hourglass, self).__init__()
        #teacher network
        self.teacher = MyResNet(block, layers, teacher_class_num)

        self.iter = 1.0
        self.annealing = annealing
        self.netstyle = netstyle


        self.get_anneal = get_anneal

        self.semodule1 = ResidualAttentionModule(in_channels=64*4, mid_channels=64*4/compress_ratio,  depth=hourglass_depth)
        self.semodule2 = ResidualAttentionModule(in_channels=128*4, mid_channels=128*4/compress_ratio, depth=hourglass_depth)
        self.semodule3 = ResidualAttentionModule(in_channels=256*4, mid_channels=256*4/compress_ratio, depth=hourglass_depth)
        self.semodule4 = ResidualAttentionModule(in_channels=512*4, mid_channels=512*4/compress_ratio, depth=hourglass_depth)


        #student network
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],student_class_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)




    def forward(self, x):

        b1_layer1, b1_layer2, b1_layer3, b1_layer4, b1 = self.teacher(x)

        b2 = self.conv1(x)
        b2 = self.bn1(b2)
        b2 = self.relu(b2)
        b2 = self.maxpool(b2)

        if self.annealing:
            annealing_value = self.get_anneal(self.iter)
        else:
            annealing_value = 1

        if self.training:
            #print "update!"
            self.iter += 1
        #print "anneal: {}".format(annealing_value)
        if self.netstyle == 0:
            b2 = self.layer1(b2) + self.semodule1(b1_layer1)*annealing_value
            b2 = self.layer2(b2) + self.semodule2(b1_layer2)*annealing_value
            b2 = self.layer3(b2) + self.semodule3(b1_layer3)*annealing_value
            b2 = self.layer4(b2) + self.semodule4(b1_layer4)*annealing_value

        elif self.netstyle == 1:
            b2 = self.layer1(b2)
            b2 = self.layer2(b2)
            b2 = self.layer3(b2)
            b2 = self.layer4(b2) + self.semodule4(b1_layer4) * annealing_value
        elif self.netstyle == 2:
            b2 = self.layer1(b2)
            b2 = self.layer2(b2)
            b2 = self.layer3(b2)+ self.semodule3(b1_layer3)*annealing_value
            b2 = self.layer4(b2) + self.semodule4(b1_layer4) * annealing_value
        elif self.netstyle == 3:
            b2 = self.layer1(b2)
            b2 = self.layer2(b2) + self.semodule2(b1_layer2)*annealing_value
            b2 = self.layer3(b2)+ self.semodule3(b1_layer3)*annealing_value
            b2 = self.layer4(b2) + self.semodule4(b1_layer4) * annealing_value
        elif self.netstyle == 4:
            b2 = self.layer1(b2) + b1_layer1*annealing_value
            b2 = self.layer2(b2) + b1_layer2*annealing_value
            b2 = self.layer3(b2) + b1_layer3*annealing_value
            b2 = self.layer4(b2) + b1_layer4*annealing_value
        elif self.netstyle == 5:
            b2 = self.layer1(b2)
            b2 = self.layer2(b2) + b1_layer2*annealing_value
            b2 = self.layer3(b2) + b1_layer3*annealing_value
            b2 = self.layer4(b2) + b1_layer4*annealing_value
        elif self.netstyle == 6:
            b2 = self.layer1(b2)
            b2 = self.layer2(b2)
            b2 = self.layer3(b2) + b1_layer3*annealing_value
            b2 = self.layer4(b2) + b1_layer4*annealing_value
        elif self.netstyle == 7:
            b2 = self.layer1(b2)
            b2 = self.layer2(b2)
            b2 = self.layer3(b2)
            b2 = self.layer4(b2) + b1_layer4*annealing_value
        elif self.netstyle == 8:
            b2 = self.layer1(b2)
            b2 = self.layer2(b2)
            b2 = self.layer3(b2) + b1_layer3*annealing_value
            b2 = self.layer4(b2)
        elif self.netstyle == 9:
            b2 = self.layer1(b2) + b1_layer1*annealing_value
            b2 = self.layer2(b2) + b1_layer2*annealing_value
            b2 = self.layer3(b2) + b1_layer3*annealing_value
            b2 = self.layer4(b2)
        else:
            raise

        b2 = self.layer5(b2)

        return b1, b2



class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels / 4, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels / 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels / 4, output_channels / 4, 3, stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels / 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_channels / 4, output_channels, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        out += residual
        return out


class ResidualAttentionModule(nn.Module):
    # input size is 112*112
    def __init__(self, in_channels, mid_channels, depth=3):
        super(ResidualAttentionModule, self).__init__()

        self.first_residual_blocks = ResidualBlock(in_channels, mid_channels)
        self.depth = depth

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, mid_channels),
            ResidualBlock(in_channels, mid_channels)
         )



        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 56*56
        self.softmax1_blocks = ResidualBlock(mid_channels, mid_channels)

        self.skip1_connection_residual_block = ResidualBlock(mid_channels, mid_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 28*28
        self.softmax2_blocks = ResidualBlock(mid_channels, mid_channels)

        self.skip2_connection_residual_block = ResidualBlock(mid_channels, mid_channels)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 14*14
        self.softmax3_blocks = ResidualBlock(mid_channels, mid_channels)
        self.skip3_connection_residual_block = ResidualBlock(mid_channels, mid_channels)
        self.mpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 7*7
        self.softmax4_blocks = nn.Sequential(
            ResidualBlock(mid_channels, mid_channels),
            ResidualBlock(mid_channels, mid_channels)
        )


        self.softmax5_blocks = ResidualBlock(mid_channels, mid_channels)
        self.softmax6_blocks = ResidualBlock(mid_channels, mid_channels)
        self.softmax7_blocks = ResidualBlock(mid_channels, mid_channels)


        self.softmax8_blocks = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, bias = False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, bias = False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(mid_channels, in_channels)

    def forward(self, x):




        # 112*112
        x = self.first_residual_blocks(x)

        out_trunk = x
        out_mpool1 = self.mpool1(x)
        # 56*56
        out_softmax1 = self.softmax1_blocks(out_mpool1)

        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        # 28*28
        out_softmax2 = self.softmax2_blocks(out_mpool2)

        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        # 14*14
        out_softmax3 = self.softmax3_blocks(out_mpool3)

        """
        out_skip3_connection = self.skip3_connection_residual_block(out_softmax3)
        out_mpool4 = self.mpool4(out_softmax3)
        # 7*7
        out_softmax4 = self.softmax4_blocks(out_mpool4)
        out_interp4 = self.interpolation4(out_softmax4) + out_softmax3
        out = out_interp4 + out_skip3_connection
        out_softmax5 = self.softmax5_blocks(out)

        """
        self.interpolation3 = nn.UpsamplingBilinear2d(size=out_softmax2.shape[2:])
        out_interp3 = self.interpolation3(out_softmax3) + out_softmax2 #change out_softmax5 to out_softmax3
        out = out_interp3 + out_skip2_connection
        out_softmax6 = self.softmax6_blocks(out)

        self.interpolation2 = nn.UpsamplingBilinear2d(size=out_softmax1.shape[2:])
        out_interp2 = self.interpolation2(out_softmax6) + out_softmax1
        out = out_interp2 + out_skip1_connection
        out_softmax7 = self.softmax7_blocks(out)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=out_trunk.shape[2:])
        out_interp1 = self.interpolation1(out_softmax7) + out_trunk
        out_softmax8 = self.softmax8_blocks(out_interp1)#normalize to 1

        #tail part
        out = (1 + out_softmax8) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


def get_handinhand(teacher_class_num, student_class_num, annealing = False,get_anneal=None, netstyle=0):
     model = HandInHandModel(block=Bottleneck, layers = [3, 4, 6, 3], teacher_class_num=teacher_class_num, student_class_num=student_class_num, annealing=annealing,get_anneal=get_anneal, netstyle=netstyle)
     return model


def get_handinhand_hourglass(teacher_class_num, student_class_num, annealing=False, get_anneal=None, netstyle=0, compress_ratio=8):
    model = HandInHandModel_Hourglass(block=Bottleneck, layers=[3, 4, 6, 3], teacher_class_num=teacher_class_num,
                            student_class_num=student_class_num, annealing=annealing, get_anneal=get_anneal,
                            netstyle=netstyle, compress_ratio = compress_ratio)

    return model

