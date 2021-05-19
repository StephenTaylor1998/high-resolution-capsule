import torch
import torch.nn as nn
import torch.nn.functional as func


def squash(s, dim=-1, eps=10e-21):
    norm = torch.norm(s, dim=dim, keepdim=True)
    return (norm / (1 + norm ** 2 + eps)) * s


class ConvertToCaps(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.unsqueeze(inputs, 2)


class FlattenCaps(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        batch, channels, dimensions, height, width = inputs.shape
        inputs = inputs.permute(0, 3, 4, 1, 2).contiguous()
        output_shape = (batch, channels * height * width, dimensions)
        return inputs.view(*output_shape)


class CapsToScalars(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.norm(inputs, dim=2)


class Conv2DCaps(nn.Module):
    def __init__(self, h, w, ch_i, n_i, ch_j, n_j, kernel_size=3, stride=1, r_num=1):
        super().__init__()
        self.ch_i = ch_i
        self.n_i = n_i
        self.ch_j = ch_j
        self.n_j = n_j
        self.kernel_size = kernel_size
        self.stride = stride
        self.r_num = r_num
        in_channels = self.ch_i * self.n_i
        out_channels = self.ch_j * self.n_j
        self.pad = 1
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.pad).cuda()

    def forward(self, inputs):
        self.batch, self.ch_i, self.n_i, self.h_i, self.w_i = inputs.shape
        x = inputs.view(self.batch, self.ch_i * self.n_i, self.h_i, self.w_i)
        x = self.conv1(x)
        width = x.shape[2]
        x = x.view(inputs.shape[0], self.ch_j, self.n_j, width, width)
        return squash(x, dim=2)


class ConvCapsLayer3D(nn.Module):
    def __init__(self, ch_i, n_i, ch_j=32, n_j=4, kernel_size=3, r_num=3):

        super().__init__()
        self.ch_i = ch_i
        self.n_i = n_i
        self.ch_j = ch_j
        self.n_j = n_j
        self.kernel_size = kernel_size
        self.r_num = r_num
        in_channels = 1
        out_channels = self.ch_j * self.n_j
        stride = (n_i, 1, 1)
        pad = (0, 1, 1)
        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=self.kernel_size,
                               stride=stride,
                               padding=pad).cuda()

    def forward(self, inputs):
        self.batch, self.ch_i, self.n_i, self.h_i, self.w_i = inputs.shape
        x = inputs.view(self.batch, self.ch_i * self.n_i, self.h_i, self.w_i)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        self.width = x.shape[-1]
        x = x.permute(0, 2, 1, 3, 4)
        x = x.view(self.batch, self.ch_i, self.ch_j, self.n_j, self.width, self.width)
        x = x.permute(0, 4, 5, 3, 2, 1).contiguous()
        self.B = x.new(x.shape[0], self.width, self.width, 1, self.ch_j, self.ch_i).zero_()
        x = self.update_routing(x, self.r_num)
        return x

    def update_routing(self, x, itr=3):

        for i in range(itr):

            tmp = self.B.permute(0, 5, 3, 1, 2, 4).contiguous().reshape(x.shape[0], self.ch_i, 1,
                                                                        self.width * self.width * self.ch_j)
            k = func.softmax(tmp, dim=-1)
            k = k.reshape(x.shape[0], self.ch_i, 1, self.width, self.width, self.ch_j).permute(0, 3, 4, 2, 5,
                                                                                               1).contiguous()
            S_tmp = k * x
            S = torch.sum(S_tmp, dim=-1, keepdim=True)
            S_hat = squash(S)

            if i < (itr - 1):
                agrements = (S_hat * x).sum(dim=3, keepdim=True)  # sum over n_j dimension
                self.B = self.B + agrements

        S_hat = S_hat.squeeze(-1)
        # batch, h_j, w_j, n_j, ch_j  = S_hat.shape
        return S_hat.permute(0, 4, 3, 1, 2).contiguous()


class Mask_CID(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target=None):
        if target is None:
            classes = torch.norm(x, dim=2)
            max_len_indices = classes.max(dim=1)[1].squeeze()
        else:
            max_len_indices = target.max(dim=1)[1]

        increasing = torch.arange(start=0, end=x.shape[0]).cuda()
        m = torch.stack([increasing, max_len_indices], dim=1)
        masked = torch.zeros((x.shape[0], 1) + x.shape[2:])
        for i in increasing:
            masked[i] = x[m[i][0], m[i][1], :].unsqueeze(0)

        return masked.squeeze(-1), max_len_indices


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules=10, num_routes=640, in_channels=8, out_channels=16, routing_iters=3):
        super().__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.routing_iters = routing_iters
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels) * 0.01)
        self.bias = nn.Parameter(torch.rand(1, 1, num_capsules, out_channels) * 0.01)

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(dim=4)
        u_hat = torch.matmul(self.W, x).squeeze()
        b_ij = x.new(x.shape[0], self.num_routes, self.num_capsules, 1).zero_()
        for itr in range(self.routing_iters):
            c_ij = func.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) + self.bias
            v_j = squash(s_j, dim=-1)
            if itr < self.routing_iters - 1:
                a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
                b_ij = b_ij + a_ij

        v_j = v_j.squeeze(1)  # .unsqueeze(-1)
        return v_j  # dim: (batch, num_capsules, out_channels or dim_capsules)

