#!/bin/bash
echo "Training..."
# ============================================== cifar 10 ============================================== #
#python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet18_dwt_tiny_half --routing-name-list Tiny_FPN

#python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet18_dwt_tiny_half --routing-name-list FPN

#python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet18_dwt_tiny_half --routing-name-list Tiny_FPN Tiny_FPN Tiny_FPN

#python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet18_dwt_tiny_half --routing-name-list FPN FPN FPN

#python train_imagenet.py -d cifar10 -a capsnet_em_routingx1 -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar

#python train_imagenet.py -d cifar10 -a capsnet_sr_routingx1 -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar

#python train_imagenet.py -d cifar10 -a capsnet_dr_routingx1 -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar

# ============================================ sub-imagenet ============================================ #
python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 100 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler imagenet --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN FPN FPN --in-shape 3 224 224


echo "done."
