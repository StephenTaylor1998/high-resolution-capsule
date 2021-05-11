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


class HRCapsTiny(nn.Module):
    def __init__(self, in_shape, num_classes=10, routing_name_list: Union[list, tuple] = None,
                 backbone=models.resnet10_tiny_half):
        super(HRCapsTiny, self).__init__()
        self.backbone = backbone(backbone=True)
        channels = self.backbone.compute_shape(in_shape)[0]
        self.primary_caps = PrimaryCaps(channels, channels, 2, 1, num_capsule=channels // 16, capsule_shape=(4, 4))
        self.routing = RoutingMatrix(channels // 16, num_classes, routing_name_list)
        self.length = LengthMatrix()

    def forward(self, x):
        x = self.backbone(x)
        x = self.primary_caps(x)
        x = self.routing(x)
        # print(x.shape)
        classes = self.length(x)
        return classes


def hr_caps_r_fpn(num_classes=10, args=None, **kwargs):
    in_shape = (3, 32, 32) if args.in_shape is None else args.in_shape
    routing_name_list = ['Tiny_FPN'] if args.routing_name_list is None else args.routing_name_list
    backbone = models.__dict__[args.backbone]
    return HRCaps(in_shape, num_classes, routing_name_list, backbone)


def hr_caps_r10_fpn(num_classes=10, args=None, **kwargs):
    in_shape = (3, 32, 32) if args.in_shape is None else args.in_shape
    routing_name_list = ['Tiny_FPN'] if args.routing_name_list is None else args.routing_name_list
    return HRCapsTiny(in_shape, num_classes, routing_name_list, models.resnet10_tiny_half)


def hr_caps_r10_dwt_fpn(num_classes=10, args=None, **kwargs):
    in_shape = (3, 32, 32) if args.in_shape is None else args.in_shape
    routing_name_list = ['Tiny_FPN'] if args.routing_name_list is None else args.routing_name_list
    return HRCapsTiny(in_shape, num_classes, routing_name_list, models.resnet10_dwt_tiny_half)


if __name__ == '__main__':
    from thop import profile

    inp = torch.ones((1, 3, 32, 32))

    # out = RoutingBlockMatrix(32, 'FPN')(inp)
    # out = RoutingBlockMatrix(32, 'Tiny_FPN')(inp)
    # out = RoutingMatrix(32, 10, ['Tiny_FPN'])(inp)
    # out = HRCaps((3, 512, 512), routing_name_list=['Tiny_FPN'])(inp)
    # out = HRCapsTiny((3, 32, 32), 10, ['FPN', 'FPN', 'FPN'], 128)(inp)
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = nn.Conv2d(3, 128, 3, 2)

        def forward(self, x):
            return self.conv(x)
    model = Model()
    # model = HRCapsTiny((3, 32, 32), 10, ['FPN', 'FPN', 'FPN'], 128)
    # print(out.shape)
    # print(out)

    macs, params = profile(model, inputs=(inp,))
    print('=' * 30)
    print(f"FLOPs: {macs / 1000000000} GFLOPs")
    print(f"Params: {params / 1000000} M")
    print('=' * 30)
