import torch
from torch import nn
from core.layers.transforms.dwt import DWTForward


class CapsConv2d(nn.Module):
    def __init__(self, dim: tuple = (4, 4), rate=1, init_mode='he_normal'):
        super(CapsConv2d, self).__init__()
        self.dim = dim
        kernel = torch.empty(dim[0], dim[1]*rate)  # (dim=dim, wave_length=4)
        kernel = nn.init.xavier_uniform_(kernel) if init_mode == 'glorot' else nn.init.kaiming_uniform_(kernel)
        self.weight = nn.Parameter(kernel)

    def forward(self, x, reshape=True):
        # [B, num_caps, dim=dim, wave_length=4, H, W] <- [B, channel, H, W]
        x = torch.reshape(x, (x.shape[0], x.shape[1] // (self.dim[0] * self.dim[1]),
                              self.dim[0], self.dim[1], x.shape[2], x.shape[3]))
        x = torch.einsum('...jikl, jm -> ...mikl', x, self.weight)
        # [B, channel, H, W] <- [B, num_caps, dim=dim, wave_length=4, H, W]
        x = torch.reshape(x, (x.shape[0], -1, x.shape[4], x.shape[5])) if reshape else x
        return x


class PrimaryDWT(nn.Module):
    def __init__(self, dim: tuple = (4, 4), wave_name='haar', init_mode='he_normal'):
        super(PrimaryDWT, self).__init__()
        self.caps_conv = CapsConv2d(dim)
        self.dwt = DWTForward(wave_name)

    def forward(self, x, reshape=True):
        # [B, channel*4, H/2, W/2] <- [B, channel, H, W]
        x = self.dwt(x)
        # [B, num_caps, dim=dim, wave_length=4, H/2, W/2] <- [B, channel*4, H/2, W/2]
        x = self.caps_conv(x, reshape)
        return x
