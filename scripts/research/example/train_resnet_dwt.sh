#!/bin/bash
echo "Training..."

python train_imagenet.py -d cifar10 -a resnet50_dwt_tiny_half -b 512 -j 32 -c 10 --epoch 400 --dist-url 'tcp://127.0.0.1:8889' \
--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar \
> data/logs/resnet50_dwt_tiny_half_cifar10.txt

python train_imagenet.py -d cifar10 -a resnet50_dwt_tiny -b 512 -j 32 -c 10 --epoch 400 --dist-url 'tcp://127.0.0.1:8889' \
--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar \
> data/logs/resnet50_dwt_tiny_cifar10.txt

python train_imagenet.py -d cifar10 -a resnet34_dwt_tiny_half -b 512 -j 32 -c 10 --epoch 400 --dist-url 'tcp://127.0.0.1:8889' \
--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar \
> data/logs/resnet34_dwt_tiny_half_cifar10.txt

python train_imagenet.py -d cifar10 -a resnet34_dwt_tiny -b 512 -j 32 -c 10 --epoch 400 --dist-url 'tcp://127.0.0.1:8889' \
--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar \
> data/logs/resnet34_dwt_tiny_cifar10.txt

python train_imagenet.py -d cifar10 -a resnet18_dwt_tiny_half -b 512 -j 32 -c 10 --epoch 400 --dist-url 'tcp://127.0.0.1:8889' \
--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar \
> data/logs/resnet18_dwt_tiny_half_cifar10.txt

python train_imagenet.py -d cifar10 -a resnet18_dwt_tiny -b 512 -j 32 -c 10 --epoch 400 --dist-url 'tcp://127.0.0.1:8889' \
--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar \
> data/logs/resnet18_dwt_tiny_cifar10.txt

echo "done."
