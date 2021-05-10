import torch
from torch import nn
from typing import Union
from core.layers.transforms.dwt import DWTForward


class PartialMatrix(nn.Module):
    def __init__(self, num_capsule: int, rate: int = 1, matrix_shape: tuple = (4, 4),
                 init_mode='glorot'):
        super(PartialMatrix, self).__init__()
        kernel = torch.empty((rate, *matrix_shape))
        kernel = nn.init.xavier_uniform_(kernel) if init_mode == 'glorot' else nn.init.kaiming_uniform_(kernel)
        self.W = nn.Parameter(kernel)
        self.middle_shape = (num_capsule, rate, *matrix_shape)

    def forward(self, x):
        x = torch.reshape(x, (*x.shape[:-3], *self.middle_shape))
        x = torch.einsum('...hijk,ikl->...hjl', x, self.W)
        return x


class GlobalMatrix(nn.Module):
    def __init__(self, num_capsule: int, matrix_shape=(4, 4), init_mode='glorot'):
        super(GlobalMatrix, self).__init__()
        kernel = torch.empty((num_capsule, *matrix_shape))
        kernel = nn.init.xavier_uniform_(kernel) if init_mode == 'glorot' else nn.init.kaiming_uniform_(kernel)
        self.W = nn.Parameter(kernel)

    def forward(self, x):
        x = torch.einsum('...ijk,hkl->...hjl', x, self.W)
        return x


class CondenseTiny(nn.Module):
    def __init__(self, num_capsule: int, rate: int = 1, matrix_shape: tuple = (4, 4),
                 init_mode='glorot'):
        super(CondenseTiny, self).__init__()
        self.rate = nn.Parameter(torch.tensor(rate, dtype=torch.float32), requires_grad=False)
        self.sparse_extraction = PartialMatrix(num_capsule, rate, matrix_shape, init_mode)
        # self.normal = nn.BatchNorm2d(num_capsule)
        self.normal = nn.LayerNorm([num_capsule, *matrix_shape])
        # self.activation = nn.ELU()
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.sparse_extraction(x)
        x = x / self.rate
        x = self.normal(x)
        x = self.activation(x)
        return x


class Condense(nn.Module):
    def __init__(self, num_capsule: int, matrix_shape=(4, 4), init_mode='glorot'):
        super(Condense, self).__init__()
        self.rate = nn.Parameter(torch.tensor(num_capsule, dtype=torch.float32), requires_grad=False)
        self.dense_extraction = GlobalMatrix(num_capsule, matrix_shape, init_mode=init_mode)
        # self.normal = nn.BatchNorm2d(num_capsule)
        self.normal = nn.LayerNorm([num_capsule, *matrix_shape])
        # self.activation = nn.ELU()
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.dense_extraction(x)
        x = x / self.rate
        x = self.normal(x)
        x = self.activation(x)
        return x


class CapsFPNTiny(nn.Module):
    def __init__(self, num_capsule: int = None, rate_list: Union[list, tuple] = None,
                 matrix_shape: Union[list, tuple] = None):
        super(CapsFPNTiny, self).__init__()
        rate_list = [2, 2, 2, 1] if rate_list is None else rate_list
        self.matrix_shape = (4, 4) if matrix_shape is None else matrix_shape
        num_capsule = num_capsule // rate_list[0]
        self.condense1 = CondenseTiny(num_capsule, rate=rate_list[0], matrix_shape=self.matrix_shape)
        num_capsule = num_capsule // rate_list[1]
        self.condense2 = CondenseTiny(num_capsule, rate=rate_list[1], matrix_shape=self.matrix_shape)
        num_capsule = num_capsule // rate_list[2]
        self.condense3 = CondenseTiny(num_capsule, rate=rate_list[2], matrix_shape=self.matrix_shape)
        num_capsule = num_capsule // rate_list[3]
        self.condense4 = CondenseTiny(num_capsule, rate=rate_list[3], matrix_shape=self.matrix_shape)

    def forward(self, x):
        l1 = self.condense1(x)
        l2 = self.condense2(l1)
        l3 = self.condense3(l2)
        l4 = self.condense4(l3)
        feature_pyramid = torch.cat([l1, l2, l3, l4], dim=-3)
        return feature_pyramid


class CapsFPN(nn.Module):
    def __init__(self, num_caps: Union[list, tuple] = None,
                 matrix_shape: Union[list, tuple] = None,):
        super(CapsFPN, self).__init__()
        num_caps = (16, 8, 4, 4) if num_caps is None else num_caps
        self.matrix_shape = (4, 4) if matrix_shape is None else matrix_shape
        self.condense1 = Condense(num_caps[0], self.matrix_shape)
        self.condense2 = Condense(num_caps[1], self.matrix_shape)
        self.condense3 = Condense(num_caps[2], self.matrix_shape)
        self.condense4 = Condense(num_caps[3], self.matrix_shape)

    def forward(self, x):
        l1 = self.condense1(x)
        l2 = self.condense2(l1)
        l3 = self.condense3(l2)
        l4 = self.condense4(l3)
        feature_pyramid = torch.cat([l1, l2, l3, l4], dim=-3)
        return feature_pyramid


class PrimaryCaps(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, down_sample_times: int,
                 num_capsule: int, capsule_shape: Union[list, tuple], out_capsule_last=True):
        super(PrimaryCaps, self).__init__()
        self.num_capsule = num_capsule
        self.capsule_shape = capsule_shape
        self.out_capsule_last = out_capsule_last
        self.down = self._make_layer(down_sample_times, in_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=1, groups=out_channel)

    def compute_shape(self, input_shape, batch_size: int = 1, data_type=torch.float32):
        inputs = torch.ones((batch_size, *input_shape), dtype=data_type)
        out = self.forward(inputs)
        return out.shape[1:]

    def _make_layer(self, down_sample_times: int, channel: int):
        layer_list = []
        for i in range(down_sample_times):
            layer_list.append(nn.Conv2d(channel, channel//4, 1))
            layer_list.append(DWTForward())
            layer_list.append(nn.BatchNorm2d(channel))

        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        x = torch.reshape(x, (x.shape[0], self.num_capsule, *self.capsule_shape, *x.shape[-2:]))
        # if out capsule last, output shape [batch, H, W, NUM_CAPS, CAPS_LENGTH]
        # if out capsule first, output shape [batch, NUM_CAPS, CAPS_LENGTH, H, W]
        x = x.permute((0, 4, 5, 1, 2, 3)) if self.out_capsule_last else x
        return x


# if __name__ == '__main__':
#     # from thop import profile
#
#     # inp = torch.ones((1, 28, 28, 16, 4, 4))
#     # out = PartialMatrix(4, rate=4)(inp)
#     # out = GlobalMatrix(32, matrix_shape=(4, 4))(inp)
#     # out = CondenseTiny(4, rate=4)(inp)
#     # out = Condense(256, matrix_shape=(4, 4))(inp)
#     # out = CapsFPNTiny(16, matrix_shape=(4, 4))(inp)
#     # out = CapsFPN([16, 8, 4, 4], matrix_shape=(4, 4))(inp)
#
#     # macs, params = profile(PrimaryCaps(256, 256, 7, 2, num_capsule=16, capsule_length=16), inputs=(inp,))
#     inp = torch.ones((1, 256, 28, 28))
#     out = PrimaryCaps(256, 256, 3, 2, 16, (4, 4))(inp)
#     print(out.shape)
#     out = GlobalMatrix(32, matrix_shape=(4, 4))(out)
#     # print(out[0, 0, 0])
#     print(out.shape)
#     # print(f"FLOPs: {macs / 1000000000} GFLOPs")
#     # print(f"Params: {params / 1000000} M")
