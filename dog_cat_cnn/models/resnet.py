# coding:utf8
from .basic_module import BasicModule
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

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
        m['conv1'] = conv1x1(inchannel, outchannel)
        m['bn1'] = nn.BatchNorm2d(outchannel)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(outchannel, outchannel, stride=stride)
        m['bn2'] = nn.BatchNorm2d(outchannel)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = conv1x1(outchannel, outchannel * 4)
        m['bn3'] = nn.BatchNorm2d(outchannel * 4)
        self.left = nn.Sequential(m)
        self.right = downsample

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(BasicModule):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inchannel = 64
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pre = nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(kernel_size=7))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, outchannel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inchannel != outchannel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inchannel, outchannel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel * block.expansion)
            )
        layers = []
        layers.append(block(self.inchannel, outchannel, stride, downsample))
        self.inchannel = outchannel * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inchannel, outchannel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.szie(0), -1)
        return self.fc(x)


def resnet18(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet18'], model_root)
        model.load_state_dict(state_dict)
    return model


def resnet34(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet34'], model_root)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'], model_root)
        model.load_state_dict(state_dict)
    return model


def resnet101(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet101'], model_root)
        model.load_state_dict(state_dict)
    return model


def resnet152(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet152'], model_root)
        model.load_state_dict(state_dict)
    return model
