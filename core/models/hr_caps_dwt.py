import torch
from torch import nn
from core.layers import operator_vector, resnet
from core.layers.layers_efficient import PrimaryCaps, Length, FCCaps
from core.models import resnet50_dwt_tiny_half
from core.layers.routing_vector import RoutingVector


class Model(nn.Module):
    def __init__(self, in_shape=None, num_classes=10):
        super(Model, self).__init__()
        if in_shape is None:
            in_shape = [3, 32, 32]
        self.backbone = resnet50_dwt_tiny_half(backbone=True)
        shape = self.backbone.compute_shape(in_shape)
        self.primary_caps = PrimaryCaps(in_channel=shape[0], out_channel=shape[0], kernel_size=shape[-1],
                                        num_capsule=shape[0]//8, capsule_length=8)
        shape = self.primary_caps.compute_shape(shape)
        self.routing = RoutingVector((shape[1], shape[0]), ['Tiny_FPN'])
        self.digit_caps = FCCaps(shape[0], shape[1], num_classes, 16)
        self.length = Length()

    def forward(self, x):
        x = self.backbone(x)
        x = self.primary_caps(x)
        x = torch.transpose(x, 1, 2)
        x = self.routing(x)
        x = torch.transpose(x, 1, 2)
        digit = self.digit_caps(x)
        classes = self.length(digit)
        return classes


def capsule_efficient_cifar(num_classes=10, args=None, **kwargs):
    in_shape = (3, 32, 32) if args.in_shape is None else args.in_shape
    return Model(in_shape, num_classes)

