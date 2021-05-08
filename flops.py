import torch

from core import models
from thop import profile
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Parameters and FLOPs Testing')
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')

    parser.add_argument('-s', '--shape', default=[1, 3, 224, 224], nargs='+', type=int,
                        help="Input image's shape.")

    parser.add_argument('-c', '--classes', default=1001, type=int, metavar='N',
                        help='number of classes (default: 1001)')

    # add on args
    parser.add_argument('--in-shape', default=(3, 32, 32), nargs='+', type=int,
                        help='Input image.')
    parser.add_argument('--pose_dim', default=4, type=int, help='Capsule pose.')
    parser.add_argument('--routing-iter', default=3, type=int, help='Capsule routing iter.')
    parser.add_argument('--capsule-arch', default=[64, 8, 16, 16, 5], nargs='+', type=int,
                        help='Capsule arch.')
    parser.add_argument('--routing-name-list', default=["FPN"], nargs='+', type=str,
                        help='FPN routing.')
    parser.add_argument('--backbone', default="resnet50_dwt_tiny_half", type=str, help='FPN routing.')

    args = parser.parse_args()

    model = models.__dict__[args.arch](num_classes=args.classes, args=args)
    print(args.shape)
    inputs = torch.randn(1, 3, 32, 32)
    macs, params = profile(model, inputs=(inputs,))
    print('='*30)
    print(f"Model Name: {args.arch}")
    print(f"FLOPs: {macs/1000000000} GFLOPs")
    print(f"Params: {params/1000000} M")
    print('=' * 30)
