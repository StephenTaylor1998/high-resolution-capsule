# ------------------------------------ classify ------------------------------------ #
# # import models from torchvision
from torchvision.models import *
# # import models from efficientnet
from .efficientnet import b0, b1, b2, b3, b4, b5, b6, b7
from .efficientnet import b0_n_channel, b1_c1, b2_c1, b3_c1, b4_c1, b5_c1, b6_c1, b7_c1

from .resnet import resnet18_cifar, resnet34_cifar, resnet50_cifar, \
    resnet18_dwt_half, resnet18_dwt_tiny, resnet18_dwt_tiny_half, resnet18_tiny_half, \
    resnet34_dwt_half, resnet34_dwt_tiny, resnet34_dwt_tiny_half, \
    resnet50_dwt_half, resnet50_dwt_tiny, resnet50_dwt_tiny_half

# ------------------------------------ capsule ------------------------------------ #

from .others.capsnet_vb import capsule_vb_mnist, capsule_vb_cifar, capsule_vb_smallnorb, capsule_vb_tiny_smallnorb

from .hr_caps_dwt import capsule_efficient_cifar, hr_caps_r_fpn

from .others import \
    capsnet_em, capsnet_em_routingx1, capsnet_em_r20_routingx1, capsnet_em_dwt_routingx1, \
    capsnet_sr, capsnet_sr_routingx1, capsnet_sr_r20_routingx1, \
    capsnet_dr, capsnet_dr_routingx1, capsnet_dr_r20_routingx1
