# ------------------------------------ classify ------------------------------------ #
# # import models from torchvision
from torchvision.models import *
# # import models from efficientnet
from .efficientnet import b0, b1, b2, b3, b4, b5, b6, b7
from .efficientnet import b0_c1, b1_c1, b2_c1, b3_c1, b4_c1, b5_c1, b6_c1, b7_c1

from .resnet import resnet18_cifar, resnet34_cifar, resnet50_cifar, \
    resnet18_dwt_half, resnet18_dwt_tiny, resnet18_dwt_tiny_half, \
    resnet34_dwt_half, resnet34_dwt_tiny, resnet34_dwt_tiny_half, \
    resnet50_dwt_half, resnet50_dwt_tiny, resnet50_dwt_tiny_half, \
    resnet10_tiny_half, resnet10_dwt_tiny_half, res_block_tiny, \
    resnet18_tiny_half, resnet18_tiny, resnet18_dwt_tiny_half_gumbel

# ------------------------------------ capsule ------------------------------------ #

from .others.capsnet_vb import capsule_vb_cifar, capsule_vb_smallnorb

from .hr_caps_dwt import hr_caps_r_fpn, lr_caps_r_fpn

from .others import \
    capsnet_dr_depthx1, capsnet_dr_depthx2, capsnet_dr_depthx3, \
    capsnet_em_depthx1, capsnet_em_depthx2, capsnet_em_depthx3, \
    capsnet_sr_depthx1, capsnet_sr_depthx2, capsnet_sr_depthx3, \
    capsnet_dp_32x32, \
    capsule_vb_smallnorb
