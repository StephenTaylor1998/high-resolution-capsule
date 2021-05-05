import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Squash(nn.Module):
    def __init__(self, eps=10e-21):
        super(Squash, self).__init__()
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True)
        return (1 - 1 / (torch.exp(n) + self.eps)) * (x / (n + self.eps))


class PrimaryCaps(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, num_capsule, capsule_length, stride=1,
                 out_capsule_last=True):
        super(PrimaryCaps, self).__init__()
        self.num_capsule = num_capsule
        self.capsule_length = capsule_length
        self.out_capsule_last = out_capsule_last
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, groups=out_channel)
        self.squash = Squash()

    def forward(self, x):
        x = self.conv(x)
        x = torch.reshape(x, (x.shape[0], self.num_capsule, self.capsule_length))
        x = self.squash(x)
        x = x.permute((0, 1, 2)) if self.out_capsule_last else x
        return x


class FCCaps(nn.Module):
    def __init__(self, in_num_capsule, in_capsule_length, out_num_capsule, out_capsule_length, init_mode='he_normal'):
        super(FCCaps, self).__init__()
        # self.out_num_capsule = out_num_capsule
        self.out_capsule_length = out_capsule_length
        kernel = torch.empty(out_num_capsule, in_num_capsule, in_capsule_length, out_capsule_length)
        kernel = nn.init.xavier_uniform_(kernel) if init_mode == 'glorot' else nn.init.kaiming_uniform_(kernel)
        self.weight = nn.Parameter(kernel)
        self.biases = nn.Parameter(torch.zeros((out_num_capsule, in_num_capsule, 1), dtype=torch.float32))
        self.factor = nn.Parameter(torch.from_numpy(np.array([self.out_capsule_length], dtype=np.float32)),
                                   requires_grad=False)

    def forward(self, x):
        u = torch.einsum('...ji,kjiz->...kjz', x, self.weight)  # u shape=(None,N,H*W*input_N,D)
        c = torch.einsum('...ij,...kj->...i', u, u)[..., None]  # b shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
        c = c / torch.sqrt(self.factor)
        c = F.softmax(c, dim=1)  # c shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
        c = c + self.biases
        s = torch.sum(torch.multiply(u, c), dim=-2)  # s shape=(None,N,D)
        v = Squash()(s)  # v shape=(None,N,D)

        return v


class Length(nn.Module):
    def __init__(self, eps=10e-21):
        super(Length, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.sqrt(torch.sum(torch.square(x), - 1) + self.eps)


class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()

    def forward(self, x, double_mask=None):
        if type(x) is list:
            if double_mask:
                x, mask1, mask2 = x
            else:
                x, mask = x
        else:
            length = torch.sqrt(torch.sum(torch.square(x), -1))
            if double_mask:
                mask1 = F.one_hot(torch.argmax(length, 1), num_classes=length.shape[1])
                mask2 = F.one_hot(torch.argmax(length, 1), num_classes=length.shape[1])

            else:

                mask = F.one_hot(torch.argmax(length, 1), num_classes=length.shape[1])

        if double_mask:
            masked1 = torch.flatten(x * torch.unsqueeze(mask1, -1), 1)
            masked2 = torch.flatten(x * torch.unsqueeze(mask2, -1), 1)
            return masked1, masked2

        else:
            masked = torch.flatten(x * torch.unsqueeze(mask, -1), 1)
            return masked


class Generator(nn.Module):
    def __init__(self, input_shape):
        super(Generator, self).__init__()
        self.input_shape = input_shape
        self.l1 = nn.Linear(160, 512)
        self.a1 = nn.ReLU(True)
        self.l2 = nn.Linear(512, 1024)
        self.a2 = nn.ReLU(True)
        self.l3 = nn.Linear(1024, np.prod(input_shape))
        self.a3 = nn.Sigmoid()

    def forward(self, x):
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        x = self.a3(self.l3(x))
        x = torch.reshape(x, (x.shape[0], *self.input_shape))
        return x


class CBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super(CBN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        return self.model(x)


class BackBoneMNIST(nn.Module):
    def __init__(self, in_channel):
        super(BackBoneMNIST, self).__init__()
        self.cbn_list = nn.Sequential(
            CBN(in_channel, 32, 5),
            CBN(32, 64, 3),
            CBN(64, 64, 3),
            CBN(64, 128, 3, 2)
        )

    def compute_shape(self, input_shape, batch_size: int = 1, data_type=torch.float32):
        inputs = torch.ones((batch_size, *input_shape), dtype=data_type)
        out = self.forward(inputs)
        return out.shape

    def forward(self, x):
        return self.cbn_list(x)


class BackBoneSmallNorb(nn.Module):
    def __init__(self, in_channel):
        super(BackBoneSmallNorb, self).__init__()
        self.cbn_list = nn.Sequential(
            CBN(in_channel, 32, 5),
            CBN(32, 64, 3),
            CBN(64, 64, 3),
            CBN(64, 128, 3, 2)
        )

    def compute_shape(self, input_shape, batch_size: int = 1):
        inputs = torch.ones((batch_size, *input_shape), dtype=torch.float32)
        out = self.cbn_list(inputs)
        return out.shape

    def forward(self, x):
        return self.cbn_list(x)
