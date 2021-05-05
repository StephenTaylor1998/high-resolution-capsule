from torch import nn
from core.layers.layers_efficient import PrimaryCaps, FCCaps, Mask, Length, Generator, BackBoneMNIST


class Capsule(nn.Module):
    def __init__(self, in_shape, decoder=True):
        super(Capsule, self).__init__()
        self.decoder = decoder
        self.cbn_list = BackBoneMNIST(in_shape[0])
        # kernel_size = int((in_shape[1] - 5 + 1 - 3 + 1 - 3 + 1 - 3 + 1) / 2)
        kernel_size = self.cbn_list.compute_shape(in_shape)
        self.primary_caps = PrimaryCaps(128, 128, kernel_size[-1], 16, 8)
        self.digit_caps = FCCaps(16, 8, 10, 16)
        self.length = Length()

    def forward(self, x):
        x = self.cbn_list(x)
        x = self.primary_caps(x)
        digit = self.digit_caps(x)
        classes = self.length(digit)
        if self.decoder:
            return digit, classes
        else:
            return classes


class Model(nn.Module):
    def __init__(self, in_shape, mode='train'):
        super(Model, self).__init__()
        self.mode = mode
        self.capsule = Capsule(in_shape[0])
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


def capsule_efficient_mnist(in_shape=(1, 28, 28), mode='train', **kwargs):
    return Model(in_shape, mode)


def capsule_efficient_smallnorb(in_shape=(2, 32, 32), mode='train', **kwargs):
    return Model(in_shape, mode)


def capsule_efficient_cifar(in_shape=(3, 32, 32), mode='train', **kwargs):
    return Model(in_shape, mode)


def capsule_efficient_without_docoder_mnist(in_shape=(1, 28, 28), **kwargs):
    return Capsule(in_shape, decoder=False)


def capsule_efficient_without_docoder_smallnorb(in_shape=(2, 32, 32), **kwargs):
    return Capsule(in_shape, decoder=False)


def capsule_efficient_without_docoder_cifar(in_shape=(3, 32, 32), **kwargs):
    return Capsule(in_shape, decoder=False)
