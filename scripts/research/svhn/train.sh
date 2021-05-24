#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo "Training..."

# ====================================== 10 k ====================================== #
python train_imagenet.py -d svhn_10k -a hr_caps_r_fpn -b 256 -j 2 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/svhn --lr-scheduler cifar --backbone resnet10_dwt_tiny_half \
--routing-name-list FPN FPN --in-shape 3 32 32 > log_capsnet_hr_fpnx2_svhn_10k.txt

python train_imagenet.py -d svhn_10k -a hr_caps_r_fpn -b 256 -j 2 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/svhn --lr-scheduler cifar --backbone resnet10_dwt_tiny_half \
--routing-name-list Tiny_FPN Tiny_FPN --in-shape 3 32 32 > log_capsnet_hr_fpnx2_svhn_10k.txt

python train_imagenet.py -d svhn_10k -a capsnet_dr_depthx2 -b 256 -j 2 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
> log_capsnet_dr_depthx2_svhn_10k.txt

python train_imagenet.py -d svhn_10k -a capsnet_em_depthx2 -b 256 -j 2 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
> log_capsnet_em_depthx2_svhn_10k.txt

python train_imagenet.py -d svhn_10k -a capsnet_sr_depthx2 -b 256 -j 2 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
> log_capsnet_sr_depthx2_svhn_10k.txt

# ====================================== 20 k ====================================== #
python train_imagenet.py -d svhn_20k -a hr_caps_r_fpn -b 256 -j 2 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/svhn --lr-scheduler cifar --backbone resnet10_dwt_tiny_half \
--routing-name-list FPN FPN --in-shape 3 32 32 > log_capsnet_hr_fpnx2_svhn_20k.txt

python train_imagenet.py -d svhn_20k -a hr_caps_r_fpn -b 256 -j 2 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/svhn --lr-scheduler cifar --backbone resnet10_dwt_tiny_half \
--routing-name-list Tiny_FPN Tiny_FPN --in-shape 3 32 32 > log_capsnet_hr_fpnx2_svhn_20k.txt

python train_imagenet.py -d svhn_20k -a capsnet_dr_depthx2 -b 256 -j 2 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
> log_capsnet_dr_depthx2_svhn_20k.txt

python train_imagenet.py -d svhn_20k -a capsnet_em_depthx2 -b 256 -j 2 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
> log_capsnet_em_depthx2_svhn_20k.txt

python train_imagenet.py -d svhn_20k -a capsnet_sr_depthx2 -b 256 -j 2 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
> log_capsnet_sr_depthx2_svhn_20k.txt

echo "done."


## ====================================== lr caps ====================================== #
#python train_imagenet.py -d svhn -a lr_caps_r_fpn -b 512 -j 1 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/svhn --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
#--routing-name-list FPN --in-shape 3 32 32 > log_capsnet_hr_fpnx1_svhn.txt
#
#python train_imagenet.py -d svhn -a lr_caps_r_fpn -b 512 -j 1 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/svhn --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
#--routing-name-list Tiny_FPN --in-shape 3 32 32 > log_capsnet_hr_tinyfpnx1_svhn.txt
#
#python train_imagenet.py -d svhn -a lr_caps_r_fpn -b 512 -j 1 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/svhn --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
#--routing-name-list FPN FPN --in-shape 3 32 32 > log_capsnet_hr_fpnx2_svhn.txt
#
#python train_imagenet.py -d svhn -a lr_caps_r_fpn -b 512 -j 1 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/svhn --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
#--routing-name-list Tiny_FPN Tiny_FPN --in-shape 3 32 32 > log_capsnet_hr_tinyfpnx2_svhn.txt
