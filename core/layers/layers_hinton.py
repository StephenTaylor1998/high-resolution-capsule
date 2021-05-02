import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Squash(nn.Module):
    def __init__(self, eps=10e-21):
        super(Squash, self).__init__()
        self.eps = eps

    def forward(self, x):
        # n = tf.norm(x, axis=-1, keepdims=True)
        # return tf.multiply(n ** 2 / (1 + n ** 2) / (n + self.eps), x)
        n = torch.norm(x, dim=1, keepdim=True)
        return torch.multiply(n ** 2 / (1 + n ** 2) / (n + self.eps), x)


class PrimaryCaps(nn.Module):
    def __init__(self, in_channel, num_capsule, capsule_length, kernel_size, stride, groups=1,
                 padding=0, out_capsule_last=True, init_mode='glorot'):
        super(PrimaryCaps, self).__init__()
        self.in_channel = in_channel
        self.num_capsule = num_capsule
        self.capsule_length = capsule_length
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.out_capsule_last = out_capsule_last
        kernel = torch.empty(self.num_capsule * self.capsule_length, self.in_channel // groups, kernel_size,
                             kernel_size)
        kernel = nn.init.kaiming_uniform_(kernel) if init_mode == 'he_normal' else nn.init.xavier_uniform_(kernel)
        self.kernel = nn.Parameter(kernel)
        self.biases = nn.Parameter(torch.zeros((self.num_capsule, self.capsule_length, 1, 1), dtype=torch.float32))
        self.squash = Squash()

    def forward(self, x):
        x = F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding, groups=self.groups)
        H, W = x.shape[-2], x.shape[-1]
        x = torch.reshape(x, (-1, self.num_capsule, self.capsule_length, H, W))
        x /= self.num_capsule
        x += self.biases
        x = self.squash(x)
        x = x.permute((0, 3, 4, 1, 2)) if self.out_capsule_last else x
        return x


class DigitCaps(nn.Module):
    def __init__(self, h, w, in_num_capsule, in_capsule_length,
                 num_capsule, capsule_length, routing=None, init_mode='glorot'):
        super(DigitCaps, self).__init__()
        self.num_capsule = num_capsule
        self.capsule_length = capsule_length
        self.routing = routing
        self.init_mode = init_mode
        weight = torch.empty((h * w * in_num_capsule, in_capsule_length, self.num_capsule * self.capsule_length))
        weight = nn.init.kaiming_uniform_(weight) if init_mode == 'he_normal' else nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        self.biases = nn.Parameter(torch.zeros((self.num_capsule, self.capsule_length), dtype=torch.float32))
        self.squash = Squash()

    def forward(self, x):
        h, w, in_num_capsule, in_capsule_length = x.shape[1:]  # input shape=(None,H,W,input_C,input_L)
        x = torch.reshape(x, (-1, h * w * in_num_capsule, in_capsule_length))  # x shape=(None,H*W*input_C,input_L)

        u = torch.einsum('...ji,jik->...jk', x, self.weight)  # u shape=(None,H*W*input_C,C*L)
        u = torch.reshape(u, (
            -1, h * w * in_num_capsule, self.num_capsule, self.capsule_length))  # u shape=(None,H*W*input_C,C,L)

        if self.routing:
            # Hinton's routing
            b = torch.zeros(u.shape[:-1])[..., None]  # b shape=(None,H*W*input_C,C,1) -> (None,i,j,1)
            for r in range(self.routing):
                c = F.softmax(b, dim=2)  # c shape=(None,H*W*input_C,C,1) -> (None,i,j,1)
                s = torch.sum(torch.multiply(u, c), dim=1, keepdim=True)  # s shape=(None,1,C,L)

                s += self.biases
                v = self.squash(s)  # v shape=(None,1,C,L)
                if r < self.routing - 1:
                    b = torch.add(b, torch.sum(torch.multiply(u, v), dim=-1, keepdim=True))
            v = v[:, 0, ...]  # v shape=(None,C,L)
        else:
            s = torch.sum(u, dim=1, keepdim=True)
            s += self.biases
            v = self.squash(s)
            v = v[:, 0, ...]
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

    def forward(self, x):
        if type(x) is list:
            x, mask = x
        else:
            x = torch.sqrt(torch.sum(torch.square(x), -1))
            indices = torch.argmax(x, 1)
            mask = F.one_hot(indices, num_classes=x.shape[1])

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
        x = torch.reshape(x, self.input_shape)
        return x