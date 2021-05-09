import torch.nn as nn
from core.layers.others.base import weights_init, resnet20_backbone
from core.layers.others.layers_dr import DynamicRouting2d, squash
from core.models import resnet18_dwt_tiny_half


class ConvNet(nn.Module):
    def __init__(self, num_classes, planes=32, num_caps=16, depth=3, backbone=resnet18_dwt_tiny_half, caps_size=16):
        super(ConvNet, self).__init__()
        self.num_caps = num_caps
        self.caps_size = caps_size
        self.depth = depth

        self.layers = backbone(backbone=True)
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # ========= ConvCaps Layers
        for d in range(1, depth):
            self.conv_layers.append(
                DynamicRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=3, stride=1, padding=1))
            nn.init.normal_(self.conv_layers[0].W, 0, 0.5)

        final_shape = 4

        # DR
        self.conv_pose = nn.Conv2d(8 * planes, num_caps * caps_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_pose = nn.BatchNorm2d(num_caps * caps_size)
        self.fc = DynamicRouting2d(num_caps, num_classes, caps_size, caps_size, kernel_size=final_shape, padding=0)
        # initialize so that output logits are in reasonable range (0.1-0.9)
        nn.init.normal_(self.fc.W, 0, 0.1)

        self.apply(weights_init)

    def forward(self, x):
        out = self.layers(x)
        # DR
        pose = self.bn_pose(self.conv_pose(out))
        b, c, h, w = pose.shape
        pose = pose.permute(0, 2, 3, 1).contiguous()
        pose = squash(pose.view(b, h, w, self.num_caps, self.caps_size))
        pose = pose.view(b, h, w, -1)
        pose = pose.permute(0, 3, 1, 2)
        for m in self.conv_layers:
            pose = m(pose)

        out = self.fc(pose)
        out = out.view(b, -1, self.caps_size)
        out = out.norm(dim=-1)
        return out


def capsnet_dr(num_classes=10, **kwargs):
    return ConvNet(num_classes)


def capsnet_dr_r20(num_classes=10, backbone=resnet20_backbone, **kwargs):
    return ConvNet(num_classes, backbone=backbone)
