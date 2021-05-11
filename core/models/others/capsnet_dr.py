import torch
import torch.nn as nn
from core import models
from core.models import resnet18_dwt_tiny_half, resnet10_tiny_half
from core.layers.others.layers_dr import DynamicRouting2d, squash
from core.layers.others.base import weights_init


class Model(nn.Module):
    def __init__(self, num_classes, planes=16, num_caps=16, depth=1, backbone=resnet18_dwt_tiny_half, caps_size=16):
        super(Model, self).__init__()
        self.num_caps = num_caps
        self.caps_size = caps_size
        self.depth = depth

        self.layers = backbone(backbone=True)
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # ========= ConvCaps Layers
        for d in range(1, depth):
            stride = 2 if d == 1 else 1
            self.conv_layers.append(
                DynamicRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=3, stride=stride, padding=1))
            nn.init.normal_(self.conv_layers[0].W, 0, 0.5)

        final_shape = 4

        # DR
        self.conv_pose = nn.Conv2d(num_caps * planes, num_caps * caps_size, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        self.bn_pose = nn.BatchNorm2d(num_caps * caps_size)
        self.fc = DynamicRouting2d(num_caps, num_classes, caps_size, caps_size, kernel_size=final_shape, padding=0)
        # initialize so that output logits are in reasonable range (0.1-0.9)
        nn.init.normal_(self.fc.W, 0, 0.1)

        self.apply(weights_init)

    def forward(self, x):
        out = self.layers(x)
        # DR
        pose = self.bn_pose(self.conv_pose(out))
        b, c, h, w = pose.shape
        pose = pose.permute(0, 2, 3, 1).contiguous()
        pose = squash(pose.view(b, h, w, self.num_caps, self.caps_size))
        pose = pose.view(b, h, w, -1)
        pose = pose.permute(0, 3, 1, 2)
        for m in self.conv_layers:
            pose = m(pose)

        out = self.fc(pose)
        out = torch.mean(out, dim=[2, 3], keepdim=False)
        out = out.view(b, -1, self.caps_size)
        out = out.norm(dim=-1)
        return out


def capsnet_dr_depthx1(num_classes=10, args=None, **kwargs):
    backbone = models.__dict__[args.backbone]
    return Model(num_classes, depth=1, backbone=backbone)


def capsnet_dr_depthx2(num_classes=10, args=None, **kwargs):
    backbone = models.__dict__[args.backbone]
    return Model(num_classes, depth=2, backbone=backbone)


def capsnet_dr_depthx3(num_classes=10, args=None, **kwargs):
    backbone = models.__dict__[args.backbone]
    return Model(num_classes, depth=3, backbone=backbone)


def capsnet_dr_r10_depth_x2(num_classes=10, backbone=resnet10_tiny_half, **kwargs):
    return Model(num_classes, backbone=backbone, depth=2)
