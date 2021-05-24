#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo "Testing..."

# ====================================== hr caps ====================================== #
python train_imagenet.py -d small_norb -a hr_caps_r_fpn -b 512 -j 10 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8890' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/smallNORB_48 --lr-scheduler cifar --backbone resnet10_tiny_half \
--routing-name-list FPN FPN --in-shape 2 32 32 \
--resume ./data/weights/hr_caps_r_fpn_epoch250_bs51_lr0.1_small_norb_FPN_FPNresnet10_tiny_half/model_best.pth.tar -t

python train_imagenet.py -d small_norb -a hr_caps_r_fpn -b 512 -j 10 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/smallNORB_48 --lr-scheduler cifar --backbone resnet10_tiny_half \
--routing-name-list FPN FPN --in-shape 2 32 32 \
--resume ./data/weights/hr_caps_r_fpn_epoch250_bs51_lr0.1_small_norb_FPN_FPNresnet10_dwt_tiny_half/model_best.pth.tar -t

# ====================================== depth 2 ====================================== #
python train_imagenet.py -d small_norb -a capsnet_dr_depthx2 -b 512 -j 10 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/smallNORB_48 --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 2 32 32 \
--resume ./data/weights/

python train_imagenet.py -d small_norb -a capsnet_em_depthx2 -b 512 -j 10 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/smallNORB_48 --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 2 32 32 \
--resume ./data/weights/

python train_imagenet.py -d small_norb -a capsnet_sr_depthx2 -b 512 -j 10 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/smallNORB_48 --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 2 32 32 \
--resume ./data/weights/

echo "done."