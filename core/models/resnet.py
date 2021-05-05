
import torch
import torch.nn as nn
from core.layers.transforms.dwt import DWTForward
from core.layers.gate_torch import DynamicGate


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(BasicBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.conv1 = nn.Conv2d(self.in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, self.expansion * self.planes, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.planes))

    def forward(self, inputs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(Bottleneck, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.conv1 = nn.Conv2d(self.in_planes, self.planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.planes)
        self.conv2 = nn.Conv2d(self.planes, self.planes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.conv3 = nn.Conv2d(self.planes, self.expansion * self.planes, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(self.expansion * self.planes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, self.expansion * self.planes, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.planes)
            )

    def forward(self, inputs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class TinyBlockDWT(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(TinyBlockDWT, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        if self.stride == 2:
            self.conv1 = DWTForward()
            self.bn1 = nn.BatchNorm2d((self.planes * 4) // 2)
            self.conv2 = nn.Conv2d((self.planes * 4) // 2, self.planes, kernel_size=1, stride=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.planes)
        else:
            self.conv1 = nn.Conv2d(self.in_planes, self.planes//2, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.planes //2)
            self.conv2 = nn.Conv2d(self.planes // 2, self.planes, kernel_size=1, stride=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.planes)

        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, self.expansion * self.planes, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.planes)
            )

    def forward(self, inputs, **kwargs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class TinyBottleDWT(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(TinyBottleDWT, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = nn.Conv2d(self.in_planes, self.planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.planes)
        if stride == 2:
            self.conv2 = DWTForward()
            self.bn2 = nn.BatchNorm2d(self.planes*8 //2)
            self.conv3 = nn.Conv2d(self.planes*8 // 2, self.expansion * self.planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion * self.planes)
        else:
            self.conv2 = nn.Conv2d(self.planes, self.planes //2, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.planes // 2)
            self.conv3 = nn.Conv2d(self.planes // 2, self.expansion * self.planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion * self.planes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, self.expansion * self.planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.planes)
            )

    def forward(self, inputs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class BasicBlockDWT(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockDWT, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.conv1 = nn.Conv2d(self.in_planes, self.planes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.planes)
        self.conv2 = nn.Conv2d(self.planes, self.planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.planes)
        if self.stride == 2:
            self.conv1 = DWTForward()
            self.bn1 = nn.BatchNorm2d(self.in_planes * 4)
            self.conv2 = nn.Conv2d(self.in_planes * 4, self.planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.planes)
        else:
            self.conv1 = nn.Conv2d(self.in_planes, self.planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.planes)
            self.conv2 = nn.Conv2d(self.planes, self.planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.planes)

        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, self.expansion * self.planes, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.planes)
            )

    def forward(self, inputs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class BottleneckDWT(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckDWT, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.conv1 = nn.Conv2d(self.in_planes, self.planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.planes)

        if self.stride == 2:
            self.conv1 = DWTForward()
            self.bn1 = nn.BatchNorm2d(self.in_planes * 4)
            self.conv2 = nn.Conv2d(self.in_planes * 4, self.planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.planes)
        else:
            self.conv1 = nn.Conv2d(self.in_planes, self.planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.planes)
            self.conv2 = nn.Conv2d(self.planes, self.planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.planes)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.conv3 = nn.Conv2d(self.planes, self.expansion * self.planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * self.planes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, self.expansion * self.planes, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.planes)
            )

    def forward(self, inputs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, half=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
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


class GumbelGateimp(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(GumbelGateimp, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.conv1 = nn.Conv2d(self.in_planes, self.planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.planes)

        if stride == 2:
            self.conv2 = nn.Conv2d(self.planes, self.planes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(self.planes, self.planes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        # self.gate = DynamicGate(self.planes)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.gate = DynamicGate(self.planes)

        self.conv3 = nn.Conv2d(self.planes, self.expansion * self.planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * self.planes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, self.expansion * self.planes, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.planes)
            )

    def forward(self, inputs):
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.gate(out, temperature=1e-3)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(inputs)
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    def __init__(self, block, num_blocks, half=True):
        super(ResNetBackbone, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        if half:
            self.layer1 = self._make_layer(block, 32, self.num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 64, self.num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 128, self.num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 256, self.num_blocks[3], stride=2)
        else:
            self.layer1 = self._make_layer(block, 64, self.num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, self.num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, self.num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, self.num_blocks[3], stride=2)

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
        return out


def resnet18_cifar(block=BasicBlock, num_blocks=None, num_classes=10, half=False, backbone=False):
    if num_blocks is None:
        num_blocks = [2, 2, 2, 2]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet34_cifar(block=BasicBlock, num_blocks=None, num_classes=10, half=False, backbone=False):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet50_cifar(block=Bottleneck, num_blocks=None, num_classes=10, half=False, backbone=False):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet18_dwt_half(block=BasicBlockDWT, num_blocks=None, num_classes=10, half=True, backbone=False):
    if num_blocks is None:
        num_blocks = [2, 2, 2, 2]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet34_dwt_half(block=BasicBlockDWT, num_blocks=None, num_classes=10, half=True, backbone=False):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet50_dwt_half(block=BottleneckDWT, num_blocks=None, num_classes=10, half=True, backbone=False):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet18_dwt_tiny(block=TinyBlockDWT, num_blocks=None, num_classes=10, half=False, backbone=False):
    if num_blocks is None:
        num_blocks = [2, 2, 2, 2]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet34_dwt_tiny(block=TinyBlockDWT, num_blocks=None, num_classes=10, half=False, backbone=False):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet50_dwt_tiny(block=TinyBottleDWT, num_blocks=None, num_classes=10, half=False, backbone=False):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet18_dwt_tiny_half(block=TinyBlockDWT, num_blocks=None, num_classes=10, half=True, backbone=False):
    if num_blocks is None:
        num_blocks = [2, 2, 2, 2]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet34_dwt_tiny_half(block=TinyBlockDWT, num_blocks=None, num_classes=10, half=True, backbone=False):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)


def resnet50_dwt_tiny_half(block=TinyBottleDWT, num_blocks=None, num_classes=10, half=True, backbone=False):
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]
    if backbone:
        return ResNetBackbone(block, num_blocks, half=half)
    else:
        return ResNet(block, num_blocks, num_classes=num_classes, half=half)