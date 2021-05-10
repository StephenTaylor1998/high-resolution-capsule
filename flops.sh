#!/bin/bash

# flops for input (3, 32, 32)
python flops.py -a hr_caps_r_fpn -s 1 3 32 32 -c 10 --backbone resnet18_dwt_tiny_half --routing-name-list Tiny_FPN Tiny_FPN Tiny_FPN --in-shape 3 224 224
python flops.py -a hr_caps_r_fpn -s 1 3 32 32 -c 10 --backbone resnet18_dwt_tiny_half --routing-name-list FPN FPN FPN --in-shape 3 224 224
python flops.py -a capsnet_em -s 1 3 32 32 -c 10
python flops.py -a capsnet_sr -s 1 3 32 32 -c 10


# flops for input (3, 224, 224)
python flops.py -a hr_caps_r_fpn -s 1 3 224 224 -c 10 --backbone resnet18_dwt_tiny_half --routing-name-list Tiny_FPN Tiny_FPN Tiny_FPN  --in-shape 3 224 224
python flops.py -a hr_caps_r_fpn -s 1 3 224 224 -c 10 --backbone resnet18_dwt_tiny_half --routing-name-list FPN FPN FPN  --in-shape 3 224 224
python flops.py -a capsnet_em -s 1 3 224 224 -c 10
python flops.py -a capsnet_sr -s 1 3 224 224 -c 10
