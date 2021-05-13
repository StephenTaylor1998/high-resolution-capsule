import torch
from torch import nn
from core import models
from typing import Union
from core.layers.operator_matrix import PrimaryCaps
from core.layers.routing_matrix import RoutingMatrix
from core.layers.routing_matrix import Length as LengthMatrix


class HRCaps(nn.Module):
    def __init__(self, in_shape, num_classes=10, routing_name_list: Union[list, tuple] = None,
                 backbone=models.resnet50_dwt_tiny_half):
        super(HRCaps, self).__init__()
        self.backbone = backbone(backbone=True, in_channel=in_shape[0])
        print(in_shape)
        shape = self.backbone.compute_shape(in_shape)
        self.primary_caps = PrimaryCaps(shape[0], shape[0], 2, 2, num_capsule=shape[0] // 16, capsule_shape=(4, 4))
        shape = self.primary_caps.compute_shape(shape)
        self.routing = RoutingMatrix(shape[2], num_classes, routing_name_list)
        self.length = LengthMatrix()

    def forward(self, x):
        x = self.backbone(x)
        x = self.primary_caps(x)
        x = self.routing(x)
        classes = self.length(x)
        return classes


class HRCapsTiny(nn.Module):
    def __init__(self, in_shape, num_classes=10, routing_name_list: Union[list, tuple] = None,
                 backbone=models.resnet10_tiny_half):
        super(HRCapsTiny, self).__init__()
        print("inshape", in_shape)
        self.backbone = backbone(backbone=True, in_channel=in_shape[0])
        channels = self.backbone.compute_shape(in_shape)[0]
        self.primary_caps = PrimaryCaps(channels, channels, 2, 1, num_capsule=channels // 16, capsule_shape=(4, 4))
        self.routing = RoutingMatrix(channels // 16, num_classes, routing_name_list)
        self.length = LengthMatrix()

    def forward(self, x):
        x = self.backbone(x)
        x = self.primary_caps(x)
        x = self.routing(x)
        classes = self.length(x)
        return classes


def hr_caps_r_fpn(num_classes=10, args=None, **kwargs):
    in_shape = (3, 32, 32) if args.in_shape is None else args.in_shape
    routing_name_list = ['Tiny_FPN'] if args.routing_name_list is None else args.routing_name_list
    backbone = models.__dict__[args.backbone]
    return HRCaps(in_shape, num_classes, routing_name_list, backbone)


def lr_caps_r_fpn(num_classes=10, args=None, **kwargs):
    in_shape = (3, 32, 32) if args.in_shape is None else args.in_shape
    routing_name_list = ['Tiny_FPN'] if args.routing_name_list is None else args.routing_name_list
    backbone = models.__dict__[args.backbone]
    return HRCapsTiny(in_shape, num_classes, routing_name_list, backbone)
