import torch
from torch import nn
from typing import Union

from core.layers.operator_matrix import CapsFPN, CapsFPNTiny, GlobalMatrix


class RoutingBlockMatrix(nn.Module):
    def __init__(self, num_capsule, routing_name: str):
        super(RoutingBlockMatrix, self).__init__()
        if routing_name == "FPN":
            self.fpn = CapsFPN((num_capsule // 2, num_capsule // 4, num_capsule // 8, num_capsule // 8))
        elif routing_name == "Tiny_FPN":
            self.fpn = CapsFPNTiny(num_capsule, rate_list=[2, 2, 2, 1])
        else:
            print(f"FPN name {routing_name} should in ['Tiny_FPN', 'FPN']")
            raise NotImplementedError
        self.norm = nn.BatchNorm2d(num_capsule)

    def forward(self, x):
        feature_pyramid = self.fpn(x)
        feature_pyramid = self.norm(feature_pyramid)
        # caps_similarity = self.caps_similarity(feature_pyramid)
        # feature_pyramid = feature_pyramid * caps_similarity
        return feature_pyramid


class RoutingMatrix(nn.Module):
    def __init__(self, num_capsule, num_classes=10, routing_name_list: Union[list, tuple] = None):
        super(RoutingMatrix, self).__init__()
        self.routings = self._make_routing(RoutingBlockMatrix, num_capsule, routing_name_list)
        self.final_mapping = GlobalMatrix(num_classes)

    def _make_routing(self, block, num_capsule, routing_name_list: list):
        layer_list = []
        for routing_name in routing_name_list:
            layer_list.append(block(num_capsule, routing_name))

        return nn.Sequential(*layer_list)

    def forward(self, x):
        feature_pyramid = self.routings(x)
        x = self.final_mapping(feature_pyramid)
        return x


class Length(nn.Module):
    def __init__(self, eps=10e-21):
        super(Length, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.mean(x, dim=[2, 3], keepdim=False)
        # return torch.sqrt(torch.sum(torch.square(x / 16.), dim=[2, 3]) + self.eps)


# if __name__ == '__main__':
#     inp = torch.ones((1, 32, 4, 4))
#
#     # out = RoutingBlockMatrix(32, 'FPN')(inp)
#     # out = RoutingBlockMatrix(32, 'Tiny_FPN')(inp)
#     # out = RoutingMatrix(32, 10, ['Tiny_FPN'])(inp)
#     out = RoutingMatrix(32, 10, ['FPN'])(inp)
#     print(out.shape)
#     print(out[0, 0, 0])
#     out = Length()(out)
#     print(out.shape)
#     print(out)
