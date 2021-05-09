import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # in sr caps (resnet-20)
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(
                    x[:, :, ::2, ::2], [0, 0, 0, 0, planes // 4, planes // 4], "constant", 0
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20_BackBone(nn.Module):
    def __init__(self, channels, num_blocks, planes=16, block=BasicBlock):
        super(ResNet20_BackBone, self).__init__()
        self.in_planes = planes
        self.conv1 = nn.Conv2d(channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        # self.layer1 = self._make_layer(block, planes, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 2 * planes, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 4 * planes, num_blocks[2], stride=2)
        self.layer1 = self._make_layer(block, planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * planes, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * planes, num_blocks[2], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def resnet20_backbone(channels=3, planes=16, block=BasicBlock, **kwargs):
    return ResNet20_BackBone(channels, [3, 3, 3], planes, block)
