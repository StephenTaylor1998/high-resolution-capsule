#!/bin/bash
echo "Training..."

#python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet18_dwt_tiny_half --routing-name-list Tiny_FPN \
#--resume ./data/weights/hr_caps_r_fpn_epoch300_bs25_lr1.0e-01_cifar10/model_best.pth.tar

#python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet18_dwt_tiny_half --routing-name-list FPN \
#--resume ./data/weights/hr_caps_r_fpn_epoch300_bs25_lr1.0e-01_cifar10/model_best.pth.tar

python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet18_dwt_tiny_half --routing-name-list Tiny_FPN Tiny_FPN Tiny_FPN

python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet18_dwt_tiny_half --routing-name-list FPN FPN FPN

python train_imagenet.py -d cifar10 -a capsnet_em -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar

python train_imagenet.py -d cifar10 -a capsnet_sr -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar

#python train_imagenet.py -d cifar10 -a capsnet_dr -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar


echo "done."
