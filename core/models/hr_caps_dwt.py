import torch
from torch import nn
from core import models
from typing import Union

from core.layers.operator_matrix import PrimaryCaps
from core.layers.routing_vector import RoutingVector
from core.layers.routing_matrix import RoutingMatrix
from core.layers.routing_matrix import Length as LengthMatrix
from core.layers.others.layers_efficient import Length as LengthVector
from core.layers.others.layers_efficient import FCCaps
from core.layers.others.layers_efficient import PrimaryCaps as PrimaryCapsEfficient


class ModelVector(nn.Module):
    def __init__(self, in_shape, num_classes=10, backbone=models.resnet50_dwt_tiny_half):
        super(ModelVector, self).__init__()
        self.backbone = backbone(backbone=True)
        shape = self.backbone.compute_shape(in_shape)
        self.primary_caps = PrimaryCapsEfficient(in_channel=shape[0], out_channel=shape[0], kernel_size=shape[-1],
                                                 num_capsule=shape[0] // 8, capsule_length=8)
        shape = self.primary_caps.compute_shape(shape)
        self.routing = RoutingVector((shape[1], shape[0]), ['Tiny_FPN'])
        self.digit_caps = FCCaps(shape[0], shape[1], num_classes, 16)
        self.length = LengthVector()

    def forward(self, x):
        x = self.backbone(x)
        x = self.primary_caps(x)
        x = torch.transpose(x, 1, 2)
        x = self.routing(x)
        x = torch.transpose(x, 1, 2)
        digit = self.digit_caps(x)
        classes = self.length(digit)
        return classes


class ModelMatrix(nn.Module):
    def __init__(self, in_shape, num_classes=10, routing_name_list: Union[list, tuple] = None,
                 backbone=models.resnet50_dwt_tiny_half):
        super(ModelMatrix, self).__init__()
        self.backbone = backbone(backbone=True)
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


def capsule_efficient_cifar(num_classes=10, args=None, **kwargs):
    in_shape = (3, 32, 32) if args.in_shape is None else args.in_shape
    return ModelVector(in_shape, num_classes)


def hr_caps_r_fpn(num_classes=10, args=None, **kwargs):
    in_shape = (3, 32, 32) if args.in_shape is None else args.in_shape
    routing_name_list = ['Tiny_FPN'] if args.routing_name_list is None else args.routing_name_list
    backbone = models.__dict__[args.backbone]
    return ModelMatrix(in_shape, num_classes, routing_name_list, backbone)


if __name__ == '__main__':
    inp = torch.ones((1, 3, 512, 512))

    # out = RoutingBlockMatrix(32, 'FPN')(inp)
    # out = RoutingBlockMatrix(32, 'Tiny_FPN')(inp)
    # out = RoutingMatrix(32, 10, ['Tiny_FPN'])(inp)
    out = ModelMatrix((3, 512, 512), routing_name_list = ['Tiny_FPN'])(inp)
    print(out.shape)
    print(out)
