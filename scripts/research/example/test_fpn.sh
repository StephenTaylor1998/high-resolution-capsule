#!/bin/bash
echo "Training..."

#python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet18_dwt_tiny_half --routing-name-list Tiny_FPN \
#--resume

python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet18_dwt_tiny_half --routing-name-list Tiny_FPN Tiny_FPN Tiny_FPN \
--resume ./data/weights/hr_caps_r_fpn_epoch300_bs25_lr0.1_cifar10_Tiny_FPN_Tiny_FPN_Tiny_FPN/model_best.pth.tar -t

python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet18_dwt_tiny_half --routing-name-list FPN FPN FPN \
--resume ./data/weights/hr_caps_r_fpn_epoch300_bs25_lr0.1_cifar10_FPN_FPN_FPN/model_best.pth.tar -t

python train_imagenet.py -d cifar10 -a capsnet_em_routingx1 -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar \
--resume ./data/weights/capsnet_em_routingx1_epoch300_bs25_lr0.1_cifar10_FPN/model_best.pth.tar -t

python train_imagenet.py -d cifar10 -a capsnet_sr_routingx1 -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar \
--resume ./data/weights/capsnet_sr_routingx1_epoch300_bs25_lr0.1_cifar10_FPN/model_best.pth.tar -t

python train_imagenet.py -d cifar10 -a capsnet_dr_routingx1 -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar \
--resume ./data/weights/capsnet_dr_routingx1_epoch300_bs25_lr0.1_cifar10_FPN/model_best.pth.tar -t
