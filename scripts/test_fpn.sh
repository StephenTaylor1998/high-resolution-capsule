#!/bin/bash
echo "Training..."

python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet18_dwt_tiny_half --routing-name-list Tiny_FPN \
-- resume
