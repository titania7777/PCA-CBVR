# original code: https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def conv3x3x3(inplanes, out_planes, stride=1):
    return nn.Conv3d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(inplanes, out_planes, stride=1):
    return nn.Conv3d(inplanes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        # 1x1x1
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        # 3x3x3
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        # 1x1x1
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
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

class ResNet(nn.Module):
    def __init__(self, block, layers, shortcut):
        super().__init__()
        self.inplanes = 64
        self.shortcut = shortcut

        # layer
        self.layer0 = nn.Sequential(
            nn.Conv3d(3, self.inplanes, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )

    def _downsample(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4))
        
        if out.is_cuda():
            zero_pads = zero_pads.cuda(out.get_device())
        
        out = torch.cat([out, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.shortcut == "A":
                downsample = partial(self._downsample, planes=planes * block.expansion, stride=stride)
            if self.shortcut == "B":
                downsample = nn.Sequential(conv1x1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample))
        
        self.inplanes = planes * block.expansion
        [layers.append(block(self.inplanes, planes)) for _ in range(1, blocks)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
        
def r3d_18(shortcut="A"):
    return ResNet(BasicBlock, [2, 2, 2, 2], shortcut)

def r3d_34(shortcut="A"):
    return ResNet(BasicBlock, [3, 4, 6, 3], shortcut)

def r3d_50(shortcut="B"):
    return ResNet(Bottleneck, [3, 4, 6, 3], shortcut)