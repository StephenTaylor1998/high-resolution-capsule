import torch
from torch import nn
from core.layers.others.layers_dp import ConvertToCaps, Conv2DCaps, ConvCapsLayer3D, FlattenCaps, \
    CapsuleLayer, CapsToScalars


class Model32x32(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=128,
                                kernel_size=3, stride=1, padding=1)
        self.batchNorm = torch.nn.BatchNorm2d(num_features=128, eps=1e-08, momentum=0.99)
        self.toCaps = ConvertToCaps()

        self.conv2dCaps1_nj_4_strd_2 = Conv2DCaps(h=32, w=32, ch_i=128, n_i=1, ch_j=32, n_j=4, kernel_size=3, stride=2,
                                                  r_num=1)
        self.conv2dCaps1_nj_4_strd_1_1 = Conv2DCaps(h=16, w=16, ch_i=32, n_i=4, ch_j=32, n_j=4, kernel_size=3, stride=1,
                                                    r_num=1)
        self.conv2dCaps1_nj_4_strd_1_2 = Conv2DCaps(h=16, w=16, ch_i=32, n_i=4, ch_j=32, n_j=4, kernel_size=3, stride=1,
                                                    r_num=1)
        self.conv2dCaps1_nj_4_strd_1_3 = Conv2DCaps(h=16, w=16, ch_i=32, n_i=4, ch_j=32, n_j=4, kernel_size=3, stride=1,
                                                    r_num=1)

        self.conv2dCaps2_nj_8_strd_2 = Conv2DCaps(h=16, w=16, ch_i=32, n_i=4, ch_j=32, n_j=8, kernel_size=3, stride=2,
                                                  r_num=1)
        self.conv2dCaps2_nj_8_strd_1_1 = Conv2DCaps(h=8, w=8, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1,
                                                    r_num=1)
        self.conv2dCaps2_nj_8_strd_1_2 = Conv2DCaps(h=8, w=8, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1,
                                                    r_num=1)
        self.conv2dCaps2_nj_8_strd_1_3 = Conv2DCaps(h=8, w=8, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1,
                                                    r_num=1)

        self.conv2dCaps3_nj_8_strd_2 = Conv2DCaps(h=8, w=8, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=2,
                                                  r_num=1)
        self.conv2dCaps3_nj_8_strd_1_1 = Conv2DCaps(h=4, w=4, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1,
                                                    r_num=1)
        self.conv2dCaps3_nj_8_strd_1_2 = Conv2DCaps(h=4, w=4, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1,
                                                    r_num=1)
        self.conv2dCaps3_nj_8_strd_1_3 = Conv2DCaps(h=4, w=4, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1,
                                                    r_num=1)

        self.conv2dCaps4_nj_8_strd_2 = Conv2DCaps(h=4, w=4, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=2,
                                                  r_num=1)
        self.conv3dCaps4_nj_8 = ConvCapsLayer3D(ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, r_num=3)
        self.conv2dCaps4_nj_8_strd_1_1 = Conv2DCaps(h=2, w=2, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1,
                                                    r_num=1)
        self.conv2dCaps4_nj_8_strd_1_2 = Conv2DCaps(h=2, w=2, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1,
                                                    r_num=1)
        self.flatCaps = FlattenCaps()
        self.digCaps = CapsuleLayer(num_capsules=10, num_routes=64 * 10, in_channels=8, out_channels=32,
                                    routing_iters=3)
        self.capsToScalars = CapsToScalars()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchNorm(x)
        x = self.toCaps(x)

        x = self.conv2dCaps1_nj_4_strd_2(x)
        x_skip = self.conv2dCaps1_nj_4_strd_1_1(x)
        x = self.conv2dCaps1_nj_4_strd_1_2(x)
        x = self.conv2dCaps1_nj_4_strd_1_3(x)
        x = x + x_skip

        x = self.conv2dCaps2_nj_8_strd_2(x)
        x_skip = self.conv2dCaps2_nj_8_strd_1_1(x)
        x = self.conv2dCaps2_nj_8_strd_1_2(x)
        x = self.conv2dCaps2_nj_8_strd_1_3(x)
        x = x + x_skip

        x = self.conv2dCaps3_nj_8_strd_2(x)
        x_skip = self.conv2dCaps3_nj_8_strd_1_1(x)
        x = self.conv2dCaps3_nj_8_strd_1_2(x)
        x = self.conv2dCaps3_nj_8_strd_1_3(x)
        x = x + x_skip
        x1 = x

        x = self.conv2dCaps4_nj_8_strd_2(x)
        x_skip = self.conv3dCaps4_nj_8(x)
        x = self.conv2dCaps4_nj_8_strd_1_1(x)
        x = self.conv2dCaps4_nj_8_strd_1_2(x)
        x = x + x_skip
        x2 = x

        xa = self.flatCaps(x1)
        xb = self.flatCaps(x2)
        x = torch.cat((xa, xb), dim=-2)
        dig_caps = self.digCaps(x)
        x = self.capsToScalars(dig_caps)
        return x


def capsnet_dp_32x32(**kwargs):
    return Model32x32()


if __name__ == '__main__':
    model = capsnet_dp_32x32().cuda()

    inp = torch.ones((1, 3, 32, 32), dtype=torch.float32).cuda()
    out = model(inp)
    print(out.shape)
