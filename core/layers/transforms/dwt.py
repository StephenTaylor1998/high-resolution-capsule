import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pywt


class DWTForward(nn.Module):
    def __init__(self, wave_name="haar"):
        super(DWTForward, self).__init__()
        wavelet = pywt.Wavelet(wave_name)
        ll = np.outer(wavelet.dec_lo, wavelet.dec_lo)
        lh = np.outer(wavelet.dec_hi, wavelet.dec_lo)
        hl = np.outer(wavelet.dec_lo, wavelet.dec_hi)
        hh = np.outer(wavelet.dec_hi, wavelet.dec_hi)
        filts = np.stack([ll[None, :, :], lh[None, :, :],
                          hl[None, :, :], hh[None, :, :]],
                         axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        channel = x.shape[1]
        filters = torch.cat([self.weight, ] * channel, dim=0)
        # in tf2 self.strides = [1, 1, 2, 2, 1]
        # x = tf.nn.conv3d(x, self.filter, padding='VALID', strides=self.strides)
        y = F.conv2d(x, filters, groups=channel, stride=2)
        return y


class DWTInverse(nn.Module):
    def __init__(self, wave_name="haar"):
        super(DWTInverse, self).__init__()
        wavelet = pywt.Wavelet(wave_name)
        ll = np.outer(wavelet.dec_lo, wavelet.dec_lo)
        lh = np.outer(wavelet.dec_hi, wavelet.dec_lo)
        hl = np.outer(wavelet.dec_lo, wavelet.dec_hi)
        hh = np.outer(wavelet.dec_hi, wavelet.dec_hi)
        filts = np.stack([ll[None, :, :], lh[None, :, :],
                          hl[None, :, :], hh[None, :, :]],
                         axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        channel = int(x.shape[1] / 4)
        filters = torch.cat([self.weight, ] * channel, dim=0)
        y = F.conv_transpose2d(x, filters, groups=channel, stride=2)
        return y
