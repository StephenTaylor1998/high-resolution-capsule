import torch
import torch.nn as nn
from core.layers.gumbel import GumbleSoftmax


class GateModule(nn.Module):
    def __init__(self, in_ch, act='relu', kernel_size=None, doubleGate=False, dwLA=False):
        super(GateModule, self).__init__()

        self.doubleGate, self.dwLA = doubleGate, dwLA
        self.inp_gs = GumbleSoftmax()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_ch = in_ch
        if act == 'relu':
            relu = nn.ReLU
        elif act == 'relu6':
            relu = nn.ReLU6
        else:
            raise NotImplementedError

        if dwLA:
            if doubleGate:
                self.inp_att = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=1, padding=0, groups=in_ch,
                              bias=True),
                    nn.BatchNorm2d(in_ch),
                    relu(inplace=True),
                    nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.Sigmoid()
                )

            self.inp_gate = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=1, padding=0, groups=in_ch, bias=True),
                nn.BatchNorm2d(in_ch),
                relu(inplace=True),
                nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(in_ch),
            )
            self.inp_gate_l = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0, groups=in_ch,
                                        bias=True)
        else:
            if doubleGate:
                reduction = 4
                self.inp_att = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch // reduction, kernel_size=1, stride=1, padding=0, bias=True),
                    relu(inplace=True),
                    nn.Conv2d(in_ch // reduction, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.Sigmoid()
                )

            self.inp_gate = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(in_ch),
                relu(inplace=True),
            )
            self.inp_gate_l = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0, groups=in_ch, bias=True)

    def forward(self, y, cb, cr, temperature=1.):
        hatten_y, hatten_cb, hatten_cr = self.avg_pool(y), self.avg_pool(cb), self.avg_pool(cr)
        hatten_d2 = torch.cat((hatten_y, hatten_cb, hatten_cr), dim=1)
        hatten_d2 = self.inp_gate(hatten_d2)
        hatten_d2 = self.inp_gate_l(hatten_d2)

        hatten_d2 = hatten_d2.reshape(hatten_d2.size(0), self.in_ch, 2, 1)
        hatten_d2 = self.inp_gs(hatten_d2, temp=temperature, force_hard=True)

        y = y * hatten_d2[:, :64, 1].unsqueeze(2)
        cb = cb * hatten_d2[:, 64:128, 1].unsqueeze(2)
        cr = cr * hatten_d2[:, 128:, 1].unsqueeze(2)

        return y, cb, cr, hatten_d2[:, :, 1]


class GateModuleA(nn.Module):
    def __init__(self, in_ch, act='relu'):
        super(GateModuleA, self).__init__()
        self.inp_gs = GumbleSoftmax()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_ch = in_ch
        if act == 'relu':
            relu = nn.ReLU
        elif act == 'relu6':
            relu = nn.ReLU6
        else:
            raise NotImplementedError

        self.inp_gate = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(in_ch),
            nn.LayerNorm([in_ch, 1, 1]),
            relu(inplace=True),
        )
        self.inp_gate_l = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0, groups=in_ch, bias=True)

    def forward(self, x, temperature=1., index=False):
        hatten = self.avg_pool(x)
        hatten_d = self.inp_gate(hatten)
        hatten_d = self.inp_gate_l(hatten_d)
        hatten_d = hatten_d.reshape(hatten_d.size(0), self.in_ch, 2, 1)
        hatten_d = self.inp_gs(hatten_d, temp=temperature, force_hard=True)
        x = x * hatten_d[:, :, 1].unsqueeze(2)
        if index:
            return x, hatten_d[:, :, 1]
        else:
            return x


if __name__ == '__main__':
    inp = torch.ones((2, 16, 224, 224))
    model = GateModuleA(in_ch=16)
    out = model(inp)
    print(out.shape)
    print(out[::, 1, 1])
