from torch import nn
from core.layers.layers_efficient import PrimaryCaps, FCCaps, Mask, Length, Generator


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


class Capsule(nn.Module):
    def __init__(self, in_channels):
        super(Capsule, self).__init__()
        self.cbn_list = nn.Sequential(
            CBN(in_channels, 32, 5),
            CBN(32, 64, 3),
            CBN(64, 64, 3),
            CBN(64, 128, 3, 2)
        )
        self.primary_caps = PrimaryCaps(128, 128, 9, 16, 8)
        self.digit_caps = FCCaps(16, 8, 10, 16)
        self.length = Length()

    def forward(self, x):
        x = self.cbn_list(x)
        x = self.primary_caps(x)

        digit = self.digit_caps(x)
        classes = self.length(digit)
        return digit, classes


class Model(nn.Module):
    def __init__(self, in_channels, out_shape, mode='train'):
        super(Model, self).__init__()
        self.mode = mode
        self.capsule = Capsule(in_channels)
        self.mask = Mask()
        self.generator = Generator(out_shape)

    def forward(self, x, y=None):
        digit, classes = self.capsule(x)
        if self.mode == "train":
            masked = self.mask([digit, y])
        else:
            masked = self.mask(digit)

        generate = self.generator(masked)
        return classes, generate


def capsule_efficient(in_channels, out_shape, mode='train'):
    return Model(in_channels, out_shape, mode)
