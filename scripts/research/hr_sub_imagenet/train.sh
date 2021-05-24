#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo "Training..."

# ====================================== hr caps ====================================== #
python train_imagenet.py -d image_folder_high_resolution -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN --in-shape 3 512 512

python train_imagenet.py -d image_folder_high_resolution -a hr_caps_r_fpn -b 64 -j 20 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN FPN --in-shape 3 512 512

# ====================================== depth 2 ====================================== #
python train_imagenet.py -d image_folder_high_resolution -a capsnet_dr_depthx2 -b 64 -j 20 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_tiny_half --in-shape 3 512 512

python train_imagenet.py -d image_folder_high_resolution -a capsnet_em_depthx2 -b 64 -j 20 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_tiny_half --in-shape 3 512 512

python train_imagenet.py -d image_folder_high_resolution -a capsnet_sr_depthx2 -b 64 -j 20 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_tiny_half --in-shape 3 512 512

echo "done."