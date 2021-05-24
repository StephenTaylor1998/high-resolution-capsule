import torch
from torch import nn
from core.layers.others.layers_dp import ConvertToCaps, Conv2DCaps, ConvCapsLayer3D, FlattenCaps, \
    CapsuleLayer, CapsToScalars, Mask_CID


class Model32x32(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_shape[0], out_channels=128,
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

    def forward(self, x, target=None):
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
        x = torch.norm(dig_caps, dim=2)
        return x


class Model28x28(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_shape[0], out_channels=128,
                                kernel_size=3, stride=1, padding=1)
        self.batchNorm = torch.nn.BatchNorm2d(num_features=128, eps=1e-08, momentum=0.99)
        self.toCaps = ConvertToCaps()

        self.conv2dCaps1_nj_4_strd_2 = Conv2DCaps(h=28, w=28, ch_i=128, n_i=1, ch_j=32, n_j=4, kernel_size=3, stride=2,
                                                  r_num=1)
        self.conv2dCaps1_nj_4_strd_1_1 = Conv2DCaps(h=14, w=14, ch_i=32, n_i=4, ch_j=32, n_j=4, kernel_size=3, stride=1,
                                                    r_num=1)
        self.conv2dCaps1_nj_4_strd_1_2 = Conv2DCaps(h=14, w=14, ch_i=32, n_i=4, ch_j=32, n_j=4, kernel_size=3, stride=1,
                                                    r_num=1)
        self.conv2dCaps1_nj_4_strd_1_3 = Conv2DCaps(h=14, w=14, ch_i=32, n_i=4, ch_j=32, n_j=4, kernel_size=3, stride=1,
                                                    r_num=1)

        self.conv2dCaps2_nj_8_strd_2 = Conv2DCaps(h=14, w=14, ch_i=32, n_i=4, ch_j=32, n_j=8, kernel_size=3, stride=2,
                                                  r_num=1)
        self.conv2dCaps2_nj_8_strd_1_1 = Conv2DCaps(h=7, w=7, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1,
                                                    r_num=1)
        self.conv2dCaps2_nj_8_strd_1_2 = Conv2DCaps(h=7, w=7, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1,
                                                    r_num=1)
        self.conv2dCaps2_nj_8_strd_1_3 = Conv2DCaps(h=7, w=7, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1,
                                                    r_num=1)

        self.conv2dCaps3_nj_8_strd_2 = Conv2DCaps(h=7, w=7, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=2,
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
        self.digCaps = CapsuleLayer(num_capsules=10, num_routes=640, in_channels=8, out_channels=16, routing_iters=3)
        self.mse_loss = nn.MSELoss(reduction="none")

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
        dig_caps = torch.norm(dig_caps, dim=2)
        return dig_caps


def capsnet_dp_32x32(args=None, **kwargs):
    in_shape = (3, 32, 32) if args.in_shape is None else args.in_shape
    return Model32x32(in_shape)


def capsnet_dp_28x28(args=None, **kwargs):
    in_shape = (1, 28, 28) if args.in_shape is None else args.in_shape
    return Model28x28(in_shape)


if __name__ == '__main__':
    class args:
        in_shape = None


    model = capsnet_dp_32x32(args).cuda()

    inp = torch.ones((10, 3, 32, 32), dtype=torch.float32).cuda()
    out = model(inp)
    print(out.shape)
    # print(out)
