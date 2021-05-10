#!/bin/bash

# ====================================== multi gpu ====================================== #
python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 100 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler imagenet --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN --in-shape 3 224 224

python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 100 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler imagenet --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN FPN FPN --in-shape 3 224 224

python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 100 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler imagenet --backbone resnet50_dwt_tiny_half \
--routing-name-list FPN --in-shape 3 224 224

python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 100 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler imagenet --backbone resnet50_dwt_tiny_half \
--routing-name-list FPN FPN FPN --in-shape 3 224 224

python train_imagenet.py -d image_folder -a resnet18_dwt_tiny_half -b 512 -j 20 -c 10 --epoch 100 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler imagenet

python train_imagenet.py -d image_folder -a resnet50_dwt_tiny_half -b 512 -j 20 -c 10 --epoch 100 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler imagenet

# ====================================== single gpu ====================================== #
