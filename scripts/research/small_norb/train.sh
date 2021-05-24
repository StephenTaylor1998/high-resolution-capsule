#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo "Training..."

# ====================================== hr caps ====================================== #
#python train_imagenet.py -d small_norb -a hr_caps_r_fpn -b 512 -j 10 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/smallNORB_48 --lr-scheduler cifar --backbone resnet10_dwt_tiny_half \
#--routing-name-list FPN FPN --in-shape 2 32 32

#python train_imagenet.py -d small_norb -a hr_caps_r_fpn -b 512 -j 10 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/smallNORB_48 --lr-scheduler cifar --backbone resnet10_tiny_half \
#--routing-name-list FPN FPN --in-shape 2 32 32

# ====================================== depth 2 ====================================== #
python train_imagenet.py -d small_norb_view_point -a capsnet_dr_depthx2 -b 128 -j 1 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/smallNORB_VP --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 1 32 32 > log-dr-smallvp.txt

python train_imagenet.py -d small_norb_view_point -a capsnet_em_depthx2 -b 512 -j 1 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/smallNORB_VP --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 1 32 32 > log-em-smallvp.txt

python train_imagenet.py -d small_norb_view_point -a capsnet_sr_depthx2 -b 128 -j 1 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/smallNORB_VP --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 1 32 32 > log-sr-smallvp.txt

echo "done."

#python train_imagenet.py -d small_norb_view_point -a hr_caps_r_fpn -b 512 -j 2 -c 10 --epoch 250 ./data/dataset/smallNORB_VP --lr-scheduler cifar --backbone res_block_tiny \
#--routing-name-list FPN FPN --in-shape 1 32 32 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0

#python train_imagenet.py -d small_norb_view_point -a hr_caps_r_fpn -b 512 -j 2 -c 10 --epoch 250 ./data/dataset/smallNORB_VP --lr-scheduler cifar --backbone res_block_tiny \
#--routing-name-list FPN FPN --in-shape 1 32 32 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0

#python train_imagenet.py -d small_norb_view_point -a capsnet_dr_depthx2 -b 512 -j 2 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/smallNORB_VP --lr-scheduler cifar --backbone res_block_tiny --in-shape 1 32 32

#python train_imagenet.py -d small_norb_view_point -a capsule_vb_smallnorb -b 512 -j 2 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/smallNORB_VP --lr-scheduler cifar --backbone res_block_tiny --in-shape 1 32 32

#python train_imagenet.py -d small_norb_view_point -a b0 -b 512 -j 1 -c 5 --epoch 250 ./data/dataset/smallNORB_VP --lr-scheduler cifar --in-shape 1 32 32 \
#--dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
