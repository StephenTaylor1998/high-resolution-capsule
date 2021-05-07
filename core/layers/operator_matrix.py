import torch
from torch import nn
from typing import Union


class PartialMatrix(nn.Module):
    def __init__(self, num_capsule: int = None, rate: int = 1, matrix_shape: tuple = (4, 4),
                 init_mode='glorot'):
        super(PartialMatrix, self).__init__()
        kernel = torch.empty((rate, *matrix_shape))
        kernel = nn.init.xavier_uniform_(kernel) if init_mode == 'glorot' else nn.init.kaiming_uniform_(kernel)
        self.W = nn.Parameter(kernel)
        self.middle_shape = (num_capsule, rate, *matrix_shape)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], *self.middle_shape))
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
        self.normal = nn.BatchNorm2d(num_capsule)
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
        self.normal = nn.BatchNorm2d(num_capsule)
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
        matrix_shape = (4, 4) if matrix_shape is None else matrix_shape
        num_capsule = num_capsule // rate_list[0]
        self.condense1 = CondenseTiny(num_capsule, rate=rate_list[0], matrix_shape=matrix_shape)
        num_capsule = num_capsule // rate_list[1]
        self.condense2 = CondenseTiny(num_capsule, rate=rate_list[1], matrix_shape=matrix_shape)
        num_capsule = num_capsule // rate_list[2]
        self.condense3 = CondenseTiny(num_capsule, rate=rate_list[2], matrix_shape=matrix_shape)
        num_capsule = num_capsule // rate_list[3]
        self.condense4 = CondenseTiny(num_capsule, rate=rate_list[3], matrix_shape=matrix_shape)

    def forward(self, x):
        l1 = self.condense1(x)
        l2 = self.condense2(l1)
        l3 = self.condense3(l2)
        l4 = self.condense4(l3)
        feature_pyramid = torch.cat([l1, l2, l3, l4], dim=1)
        return feature_pyramid


class CapsFPN(nn.Module):
    def __init__(self, num_caps: Union[list, tuple] = None,
                 matrix_shape: Union[list, tuple] = None,):
        super(CapsFPN, self).__init__()
        num_caps = (16, 8, 4, 4) if num_caps is None else num_caps
        matrix_shape = (4, 4) if matrix_shape is None else matrix_shape
        self.condense1 = Condense(num_caps[0], matrix_shape)
        self.condense2 = Condense(num_caps[1], matrix_shape)
        self.condense3 = Condense(num_caps[2], matrix_shape)
        self.condense4 = Condense(num_caps[3], matrix_shape)
        # self.feature_pyramid = layers.Concatenate(axis=-3)

    def forward(self, x):
        l1 = self.condense1(x)
        l2 = self.condense2(l1)
        l3 = self.condense3(l2)
        l4 = self.condense4(l3)
        feature_pyramid = torch.cat([l1, l2, l3, l4], dim=1)
        return feature_pyramid


# if __name__ == '__main__':
#     inp = torch.ones((1, 16, 4, 4))
#     # out = PartialMatrix(4, rate=4)(inp)
#     # out = GlobalMatrix(8, matrix_shape=(4, 4))(inp)
#     # out = CondenseTiny(4, rate=4)(inp)
#     # out = Condense(256, matrix_shape=(4, 4))(inp)
#     # out = CapsFPNTiny(16, matrix_shape=(4, 4))(inp)
#     out = CapsFPN([16, 8, 4, 4], matrix_shape=(4, 4))(inp)
#     print(out.shape)
#     print(out[0, 0, 0])
