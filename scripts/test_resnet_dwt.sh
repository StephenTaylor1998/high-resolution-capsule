#!/bin/bash
echo "Testing..."

#python train_imagenet.py -d cifar10 -a resnet50_dwt_tiny_half -b 512 -j 32 -c 10 --dist-url 'tcp://127.0.0.1:8889' \
#--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar -t \
#--resume ./data/weights/resnet50_dwt_tiny_half_epoch400_bs51_lr1.0e-01_cifar10/checkpoint_epoch399.pth.tar\
#> data/logs/resnet50_dwt_tiny_half_cifar10.txt
#
#python train_imagenet.py -d cifar10 -a resnet50_dwt_tiny -b 512 -j 32 -c 10 --dist-url 'tcp://127.0.0.1:8889' \
#--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar -t \
#--resume ./data/weights/resnet50_dwt_tiny_epoch400_bs51_lr1.0e-01_cifar10/checkpoint_epoch399.pth.tar\
#> data/logs/resnet50_dwt_tiny_cifar10.txt

#python train_imagenet.py -d cifar10 -a resnet34_dwt_tiny_half -b 512 -j 32 -c 10 --dist-url 'tcp://127.0.0.1:8889' \
#--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar -t \
#--resume ./data/weights/resnet34_dwt_tiny_half_epoch400_bs51_lr1.0e-01_cifar10/checkpoint_epoch399.pth.tar\
#> data/logs/resnet34_dwt_tiny_half_cifar10.txt

python train_imagenet.py -d cifar10 -a resnet34_dwt_tiny -b 512 -j 32 -c 10 --dist-url 'tcp://127.0.0.1:8889' \
--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar -t \
--resume ./data/weights/resnet34_dwt_tiny_epoch400_bs51_lr1.0e-01_cifar10/checkpoint_epoch399.pth.tar\
> data/logs/resnet34_dwt_tiny_cifar10.txt
#
#python train_imagenet.py -d cifar10 -a resnet18_dwt_tiny_half -b 512 -j 32 -c 10 --dist-url 'tcp://127.0.0.1:8889' \
#--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar -t \
#--resume ./data/weights/resnet18_dwt_tiny_half_epoch400_bs51_lr1.0e-01_cifar10/checkpoint_epoch399.pth.tar\
#> data/logs/resnet18_dwt_tiny_half_cifar10.txt
#
#python train_imagenet.py -d cifar10 -a resnet18_dwt_tiny -b 512 -j 32 -c 10 --dist-url 'tcp://127.0.0.1:8889' \
#--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar -t \
#--resume ./data/weights/resnet18_dwt_tiny_epoch400_bs51_lr1.0e-01_cifar10/checkpoint_epoch399.pth.tar\
#> data/logs/resnet18_dwt_tiny_cifar10.txt

echo "done."
