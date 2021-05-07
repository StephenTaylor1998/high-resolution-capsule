# ------------------------------------ classify ------------------------------------ #
# # import models from torchvision
from torchvision.models import *
# # import models from efficientnet
from .efficientnet import b0, b1, b2, b3, b4, b5, b6, b7
from .efficientnet import b0_n_channel, b1_c1, b2_c1, b3_c1, b4_c1, b5_c1, b6_c1, b7_c1

from .resnet import resnet18_cifar, resnet34_cifar, resnet50_cifar, \
    resnet18_dwt_half, resnet18_dwt_tiny, resnet18_dwt_tiny_half, \
    resnet34_dwt_half, resnet34_dwt_tiny, resnet34_dwt_tiny_half, \
    resnet50_dwt_half, resnet50_dwt_tiny, resnet50_dwt_tiny_half

# ------------------------------------ capsule ------------------------------------ #
from .hinton import \
    capsule_hinton_cifar, capsule_hinton_without_docoder_cifar, \
    capsule_hinton_nmist, capsule_hinton_without_docoder_mnist, \
    capsule_hinton_smallnorb, capsule_hinton_without_docoder_smallnorb

from .efficient import \
    capsule_efficient_mnist, capsule_efficient_without_docoder_mnist, \
    capsule_efficient_cifar, capsule_efficient_without_docoder_cifar, \
    capsule_efficient_smallnorb, capsule_efficient_without_docoder_smallnorb

from .vb_capsnet import capsule_vb_mnist, capsule_vb_cifar, capsule_vb_smallnorb, capsule_vb_tiny_smallnorb

from .hr_caps_dwt import capsule_efficient_cifar, hr_caps_r_fpn
