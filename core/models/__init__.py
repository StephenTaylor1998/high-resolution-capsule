# ------------------------------------ classify ------------------------------------ #
# # import models from torchvision
from torchvision.models import *
# # import models from efficientnet
# from .efficientnet import b0, b1, b2, b3, b4, b5, b6, b7
# from .efficientnet import b0_n_channel, b1_c1, b2_c1, b3_c1, b4_c1, b5_c1, b6_c1, b7_c1

# ------------------------------------ capsule ------------------------------------ #
from .hinton import capsule_hinton, capsule_hinton_without_docoder
from .efficient import \
    capsule_efficient_28x28x1, \
    capsule_efficient_32x32x2, \
    capsule_efficient_32x32x3, \
    capsule_efficient_without_docoder_28x28x1, \
    capsule_efficient_without_docoder_32x32x2, \
    capsule_efficient_without_docoder_32x32x3
from .backone_torch import resnet18_cifar, resnet34_cifar, resnet50_cifar, resnet101_cifar, resnet152_cifar,\
    resnet50_cifar_dwt_half,resnet18_cifar_dwt_half,resnet34_cifar_dwt_half,\
    resnet34_cifar_dwt_tiny_half,resnet18_cifar_dwt_tiny_half,resnet50_cifar_dwt_tiny_half,\
    resnet18_cifar_dwt_tiny,resnet34_cifar_dwt_tiny,resnet50_cifar_dwt_tiny


