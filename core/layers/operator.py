import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional


class Condense(nn.Module):

    def __init__(self, in_channels, scale_rate=1):
        super(Condense, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                              kernel_size=scale_rate, stride=scale_rate, bias=False)
        self.bn = nn.BatchNorm1d(in_channels)
        # self.activation = nn.Hardswish()
        # self.activation = nn.Sigmoid()
        # self.activation = nn.Tanh()

    def compute_shape(self, input_shape, batch_size: int = 1, data_type=torch.float32):
        inputs = torch.ones((batch_size, *input_shape), dtype=data_type)
        out = self.forward(inputs)
        return out.shape[1:]

    def forward(self, x):
        x = self.bn(self.conv(x))
        # x = self.activation(x)
        return x


class FPN(nn.Module):
    def __init__(self, in_shape, scale_rate_list=None):
        super(FPN, self).__init__()
        scale_rate_list = [2, 2, 2, 1] if scale_rate_list is None else scale_rate_list
        self.condense1 = Condense(in_shape[0], scale_rate_list[0])
        shape1 = self.condense1.compute_shape(in_shape)
        self.condense2 = Condense(shape1[0], scale_rate_list[1])
        shape2 = self.condense2.compute_shape(shape1)
        self.condense3 = Condense(shape2[0], scale_rate_list[2])
        shape3 = self.condense3.compute_shape(shape2)
        self.condense4 = Condense(shape3[0], scale_rate_list[3])
        self.feature_pyramid = torch.cat

    def forward(self, x):
        l1 = self.condense1(x)
        l2 = self.condense2(l1)
        l3 = self.condense3(l2)
        l4 = self.condense4(l3)
        pyramid = self.feature_pyramid([l1, l2, l3, l4], dim=2)
        return pyramid


class CapsSimilarity(nn.Module):
    def __init__(self):
        super(CapsSimilarity, self).__init__()

    def forward(self, x, grad=False):
        # x = x if grad else Variable(x.data.clone())
        global_center = torch.sum(x, dim=2, keepdim=True)
        x = torch.mul(x, global_center)
        x = torch.mean(x, dim=1, keepdim=True)
        return x


if __name__ == '__main__':
    # inp = torch.ones((2, 8, 16), dtype=torch.float32)
    inp = torch.randn((2, 8, 16), dtype=torch.float32)
    # out = FPN((8, 16))(inp)
    out = CapsSimilarity()(inp)
    print(out.shape)
    print(out)
