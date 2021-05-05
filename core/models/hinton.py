from torch import nn
from core.layers.layers_hinton import PrimaryCaps, DigitCaps, Mask, Length, Generator


class Capsule(nn.Module):
    def __init__(self, in_shape, num_classes, decoder=True):
        super(Capsule, self).__init__()
        self.decoder = decoder
        self.conv = nn.Conv2d(in_shape[0], 256, kernel_size=9)
        self.bn = nn.BatchNorm2d(256)
        self.primary_caps = PrimaryCaps(256, 32, 8, 9, 2)
        h = (in_shape[2] - 9 + 1 - 9 + 1)//2
        w = (in_shape[3] - 9 + 1 - 9 + 1)//2
        self.digit_caps = DigitCaps(h, w, 32, 8, num_classes, 16, routing=3)
        self.length = Length()

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.primary_caps(x)
        digit = self.digit_caps(x)
        classes = self.length(digit)
        if self.decoder:
            return digit, classes
        else:
            return classes


class Model(nn.Module):
    def __init__(self, in_shape, num_classes, mode='train'):
        super(Model, self).__init__()
        self.mode = mode
        self.capsule = Capsule(in_shape, num_classes)
        self.mask = Mask()
        self.generator = Generator(in_shape)

    def forward(self, x, y=None):
        digit, classes = self.capsule(x)
        if self.mode == "train":
            masked = self.mask([digit, y])
        else:
            masked = self.mask(digit)

        generate = self.generator(masked)
        return classes, generate


def capsule_hinton_nmist(num_classes=10, args=None, **kwargs):
    in_shape = (1, 28, 28) if args.in_shape is None else args.in_shape
    mode = 'train' if args.mode is None else args.mode
    return Model(in_shape, num_classes, mode)


def capsule_hinton_smallnorb(num_classes=5, args=None, **kwargs):
    in_shape = (2, 32, 32) if args.in_shape is None else args.in_shape
    mode = 'train' if args.mode is None else args.mode
    return Model(in_shape, num_classes, mode)


def capsule_hinton_cifar(num_classes=10, args=None, **kwargs):
    in_shape = (3, 32, 32) if args.in_shape is None else args.in_shape
    mode = 'train' if args.mode is None else args.mode
    return Model(in_shape, num_classes, mode)


def capsule_hinton_without_docoder_mnist(num_classes=10, args=None, **kwargs):
    in_shape = (1, 28, 28) if args.in_shape is None else args.in_shape
    return Capsule(in_shape, num_classes, decoder=False)


def capsule_hinton_without_docoder_smallnorb(num_classes=5, args=None, **kwargs):
    in_shape = (2, 32, 32) if args.in_shape is None else args.in_shape
    return Capsule(in_shape, num_classes, decoder=False)


def capsule_hinton_without_docoder_cifar(num_classes=10, args=None, **kwargs):
    in_shape = (3, 32, 32) if args.in_shape is None else args.in_shape
    return Capsule(in_shape, num_classes, decoder=False)