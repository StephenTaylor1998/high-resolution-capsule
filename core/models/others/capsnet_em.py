import torch
import torch.nn as nn

from core import models
from core.layers.others.base import weights_init, resnet20_backbone
from core.layers.others.layers_em import EmRouting2d
from core.models import resnet18_dwt_tiny_half, resnet18_tiny_half, resnet10_tiny_half


class Model(nn.Module):
    def __init__(self, num_classes, planes=16, num_caps=16, depth=3, backbone=resnet18_dwt_tiny_half, caps_size=16):
        super(Model, self).__init__()
        self.num_caps = num_caps
        self.depth = depth
        self.layers = backbone(backbone=True)
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        # ========= ConvCaps Layers
        for d in range(1, depth):
            stride = 2 if d == 1 else 1
            self.conv_layers.append(EmRouting2d(num_caps, num_caps, caps_size, kernel_size=3, stride=stride, padding=1))
            self.norm_layers.append(nn.BatchNorm2d(4 * 4 * num_caps))

        final_shape = 4

        # EM
        self.conv_a = nn.Conv2d(num_caps * planes, num_caps, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_pose = nn.Conv2d(num_caps * planes, num_caps * caps_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(num_caps)
        self.bn_pose = nn.BatchNorm2d(num_caps * caps_size)
        self.fc = EmRouting2d(num_caps, num_classes, caps_size, kernel_size=final_shape, padding=0)

        self.apply(weights_init)

    def forward(self, x):
        out = self.layers(x)

        # EM
        a, pose = self.conv_a(out), self.conv_pose(out)
        a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)

        for m, bn in zip(self.conv_layers, self.norm_layers):
            a, pose = m(a, pose)
            pose = bn(pose)

        a, _ = self.fc(a, pose)
        out = torch.mean(a, dim=[2, 3], keepdim=False)
        return out


def capsnet_em_depthx1(num_classes=10, args=None, **kwargs):
    backbone = models.__dict__[args.backbone]
    return Model(num_classes, depth=1, backbone=backbone)


def capsnet_em_depthx2(num_classes=10, args=None, **kwargs):
    backbone = models.__dict__[args.backbone]
    return Model(num_classes, depth=2, backbone=backbone)


def capsnet_em_depthx3(num_classes=10, args=None, **kwargs):
    backbone = models.__dict__[args.backbone]
    return Model(num_classes, depth=3, backbone=backbone)


def capsnet_em_r18dwt_depthx1(num_classes=10, **kwargs):
    return Model(num_classes, depth=1, backbone=resnet18_dwt_tiny_half)


def capsnet_em_r18dwt_depthx2(num_classes=10, **kwargs):
    return Model(num_classes, depth=2, backbone=resnet18_dwt_tiny_half)


def capsnet_em_r10_depth_x2(num_classes=10, backbone=resnet10_tiny_half, **kwargs):
    return Model(num_classes, backbone=backbone, depth=2)
