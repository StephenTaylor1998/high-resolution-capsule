import torch
from torch import nn


class Block2Channel3d(nn.Module):
    """
    #     example:
    #     block_shape=(2, 2)
    #     output_channel_first=False
    #     input=
    #             /\-------------/\
    #           / 1 \----------/-5 \
    #         / 3  2 \-------/-7--6 \
    #       / 1  4  1 \----/-5--8--5 \
    #     / 3  2  3  2/--/-7--6--7--6/
    #     \  4  1  4/----\--8--5--8/
    #      \  3  2/-------\--7--6/
    #       \  4/----------\--8/
    #        \/-------------\/
    #
    #     output=
    #         /\------\--------\
    #       / 1 \2 3 4 \5 6 7 8 \
    #     / 1  1/ 2 3 4/ 5 6 7 8/
    #     \  1/ 2 3 4/ 5 6 7 8/
    #      \/------/--------/
    #     :param block_shape: (block_h, block_w) example (2, 2),
    #     block shape should <= input tensor shape
    #     :param output_channel_first: channel first ==>> True & channel last ==>> False;
    #     :param check_shape: check shape before operator
    #     :return: [batch, h//block_h, w//block_w, num_caps * block_h * block_w, 4, 4]
    #     """
    def __init__(self, block_shape):
        super(Block2Channel3d, self).__init__()
        self.block_h = block_shape[0]
        self.block_w = block_shape[-1]

    def forward(self, x):
        shape = x.shape  # [b, h, w, n, 4, 4]
        x = torch.reshape(x, (shape[0],
                              shape[1] // self.block_h, self.block_h,
                              shape[2] // self.block_w, self.block_w,
                              *shape[3:])
                          )
        # x = x.permute(0, 2, 4, 5, 6, 7, 1, 3)
        x = x.permute(0, 1, 3, 2, 4, 5, 6, 7)
        x = torch.reshape(x, (shape[0],
                              shape[1] // self.block_h,
                              shape[2] // self.block_w,
                              shape[3] * self.block_h * self.block_w,
                              *shape[4:])
                          )
        return x
