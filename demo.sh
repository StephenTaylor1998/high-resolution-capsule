#!/bin/bash

python train_imagenet.py -d cifar10 -a capsnet_em -b 512 -j 32 -c 10 --epoch 400 \
./data/dataset/ --lr-scheduler cifar

python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' \
--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar \
--backbone resnet50_dwt_tiny_half --routing-name-list Tiny_FPN

python flops.py -a hr_caps_r_fpn -s 1 3 32 32 -c 10 --backbone resnet50_dwt_tiny_half --routing-name-list Tiny_FPN