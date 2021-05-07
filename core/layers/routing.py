import torch
from torch import nn

from core.layers.operator import FPN, CapsSimilarity


class RoutingBlockVector(nn.Module):
    def __init__(self, in_shape, routing_name):
        super(RoutingBlockVector, self).__init__()
        if routing_name == "Tiny_FPN":
            self.fpn = FPN(in_shape, [2, 2, 2, 1])
        elif routing_name == "FPN":
            raise NotImplementedError
        else:
            print(f"FPN name {routing_name} should in ['Tiny_FPN', 'FPN']")
            raise NotImplementedError

        # self.similarity = CapsSimilarity()
        self.bn = nn.BatchNorm1d(in_shape[0])

    def forward(self, x):
        feature_pyramid = self.fpn(x)
        # caps_similarity = self.similarity(x)
        # feature_pyramid = feature_pyramid * caps_similarity
        feature_pyramid = self.bn(feature_pyramid)
        return feature_pyramid


class RoutingVector(nn.Module):
    def __init__(self, in_shape, routing_name_list):
        super(RoutingVector, self).__init__()
        self.routings = self._make_routing(RoutingBlockVector, in_shape, routing_name_list)

    def _make_routing(self, block, in_shape, routing_name_list: list):
        layer_list = []
        for routing_name in routing_name_list:
            layer_list.append(block(in_shape, routing_name))

        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.routings(x)
        return x


if __name__ == '__main__':
    # inp = torch.ones((2, 8, 16), dtype=torch.float32)
    inp = torch.randn((2, 8, 16), dtype=torch.float32)
    # out = RoutingBlockVector((8, 16), routing_name='Tiny_FPN')(inp)
    out = RoutingVector((8, 16), ['Tiny_FPN', 'Tiny_FPN'])(inp)
    print(out.shape)
    print(out)