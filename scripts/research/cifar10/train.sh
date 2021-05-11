#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo "Training..."

# ====================================== hr caps ====================================== #
python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half \
--routing-name-list FPN --in-shape 3 32 32

python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half \
--routing-name-list FPN FPN --in-shape 3 32 32

python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half \
--routing-name-list FPN FPN FPN --in-shape 3 32 32

# ====================================== depth 1 ====================================== #
python train_imagenet.py -d cifar10 -a capsnet_dr_depthx1 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32

python train_imagenet.py -d cifar10 -a capsnet_em_depthx1 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32

python train_imagenet.py -d cifar10 -a capsnet_sr_depthx1 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32

# ====================================== depth 2 ====================================== #
python train_imagenet.py -d cifar10 -a capsnet_dr_depthx2 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32

python train_imagenet.py -d cifar10 -a capsnet_em_depthx2 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32

python train_imagenet.py -d cifar10 -a capsnet_sr_depthx2 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32

# ====================================== depth 3 ====================================== #
python train_imagenet.py -d cifar10 -a capsnet_dr_depthx3 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32

python train_imagenet.py -d cifar10 -a capsnet_em_depthx3 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32

python train_imagenet.py -d cifar10 -a capsnet_sr_depthx3 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32

echo "done."