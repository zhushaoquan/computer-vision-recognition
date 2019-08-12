# coding:utf8
from .basic_module import BasicModule
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


'''
resnet 18,34 using resudual 
network: conv(3x3x64)-->relu-->conv(3x3x64)
'''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannel, outchannel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(inchannel, outchannel, stride)
        m['bn1'] = nn.BatchNorm2d(outchannel)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(outchannel, outchannel)
        m['bn2'] = nn.BatchNorm2d(outchannel)
        self.left = nn.Sequential(m)
        self.right = downsample

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


'''
resnet 50,101,150 using resudual 
network: conv(1x1x64)-->relu-->conv(3x3x64)-->relu-->conv(1x1x256)
'''


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inchannel, outchannel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv1x1(inchannel, outchannel, stride)
        m['bn1'] = nn.BatchNorm2d(outchannel)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(outchannel, outchannel, stride)
        m['bn2'] = nn.BatchNorm2d(outchannel)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = conv1x1(outchannel, outchannel * 4, stride)
        m['bn3'] = nn.BatchNorm2d(outchannel * 4)
        self.left = nn.Sequential(m)
        self.right = downsample

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

