import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, dropout, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.dropout = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != planes:
            self.residual = conv1x1(inplanes, planes, stride)
            self.use_conv1x1 = True
        else:
            self.use_conv1x1 = False

    def forward(self, x):
        if self.use_conv1x1:
            residual = self.residual(x)
        else:
            residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out += residual

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, width, num_classes=10, dropout=0.3):
        super(WideResNet, self).__init__()

        layer = (depth - 4) // 6

        self.inplanes = 16
        self.conv = conv3x3(3, 16)
        self.layer1 = self._make_layer(16*width, layer, dropout)
        self.layer2 = self._make_layer(32*width, layer, dropout, stride=2)
        self.layer3 = self._make_layer(64*width, layer, dropout, stride=2)
        self.bn = nn.BatchNorm2d(64*width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64*width, num_classes)

    def _make_layer(self, planes, blocks, dropout, stride=1):
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, dropout, stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, dropout, stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
