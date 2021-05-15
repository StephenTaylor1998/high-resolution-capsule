import torch
import torch.nn as nn

from core.layers.resnet import \
    BasicBlock, \
    Bottleneck, \
    TinyBasicBlock, \
    BasicBlockDWT, \
    BottleneckDWT, \
    TinyBlockDWT, \
    TinyBottleDWT, \
    ResNetBackbone, \
    ResBlockLR


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, half=True, in_channel=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        if half:
            self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        else:
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(self.in_planes, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layer_list = []
        for stride in strides:
            layer_list.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layer_list)

    def forward(self, inputs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)
        out = self.softmax(out)
        return out


def resnet18_cifar(block=BasicBlock, num_blocks=None, num_classes=10,
                   half=False, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [2, 2, 2, 2]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half, in_channel=in_channel)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half, in_channel=in_channel)


def resnet34_cifar(block=BasicBlock, num_blocks=None, num_classes=10,
                   half=False, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half, in_channel=in_channel)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half, in_channel=in_channel)


def resnet50_cifar(block=Bottleneck, num_blocks=None, num_classes=10,
                   half=False, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half, in_channel=in_channel)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half, in_channel=in_channel)


def resnet18_dwt_half(block=BasicBlockDWT, num_blocks=None, num_classes=10,
                      half=True, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [2, 2, 2, 2]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half, in_channel=in_channel)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half, in_channel=in_channel)


def resnet34_dwt_half(block=BasicBlockDWT, num_blocks=None, num_classes=10,
                      half=True, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half, in_channel=in_channel)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half, in_channel=in_channel)


def resnet50_dwt_half(block=BottleneckDWT, num_blocks=None, num_classes=10,
                      half=True, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half, in_channel=in_channel)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half, in_channel=in_channel)


def resnet18_dwt_tiny(block=TinyBlockDWT, num_blocks=None, num_classes=10,
                      half=False, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [2, 2, 2, 2]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half, in_channel=in_channel)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half, in_channel=in_channel)


def resnet34_dwt_tiny(block=TinyBlockDWT, num_blocks=None, num_classes=10,
                      half=False, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half, in_channel=in_channel)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half, in_channel=in_channel)


def resnet50_dwt_tiny(block=TinyBottleDWT, num_blocks=None, num_classes=10,
                      half=False, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half, in_channel=in_channel)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half, in_channel=in_channel)


def resnet18_dwt_tiny_half(block=TinyBlockDWT, num_blocks=None, num_classes=10,
                           half=True, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [2, 2, 2, 2]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half, in_channel=in_channel)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half, in_channel=in_channel)


def resnet34_dwt_tiny_half(block=TinyBlockDWT, num_blocks=None, num_classes=10,
                           half=True, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half, in_channel=in_channel)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half, in_channel=in_channel)


def resnet50_dwt_tiny_half(block=TinyBottleDWT, num_blocks=None, num_classes=10,
                           half=True, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half, in_channel=in_channel)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half, in_channel=in_channel)


def resnet18_tiny_half(block=TinyBasicBlock, num_blocks=None, num_classes=10,
                       half=True, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [2, 2, 2, 2]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half, in_channel=in_channel)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half, in_channel=in_channel)


def resnet18_tiny(block=TinyBasicBlock, num_blocks=None, num_classes=10,
                  half=False, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [2, 2, 2, 2]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half, in_channel=in_channel)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half, in_channel=in_channel)


def resnet10_tiny_half(block=TinyBasicBlock, num_blocks=None, num_classes=10,
                       half=True, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]
    if backbone:
        return ResBlockLR(block, num_blocks, half=half, in_channel=in_channel)
    else:
        raise NotImplemented


def resnet10_dwt_tiny_half(block=TinyBlockDWT, num_blocks=None, num_classes=10,
                           half=True, backbone=False, in_channel=3, **kwargs):
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]
    if backbone:
        return ResBlockLR(block, num_blocks, half=half, in_channel=in_channel)
    else:
        raise NotImplemented
