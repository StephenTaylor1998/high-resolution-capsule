from torch import nn
from core.layers.layers_hinton import PrimaryCaps, DigitCaps, Mask, Length, Generator


class Capsule(nn.Module):
    def __init__(self, in_channels):
        super(Capsule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=9)
        self.bn = nn.BatchNorm2d(256)
        self.primary_caps = PrimaryCaps(256, 32, 8, 9, 2)
        self.digit_caps = DigitCaps(6, 6, 32, 8, 10, 16, routing=3)
        self.length = Length()

    def forward(self, x):
        x = self.bn(self.conv(x))
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
