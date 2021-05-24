#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo "Training..."

# ====================================== hr caps ====================================== #
# * Acc@1 94.400 Acc@5 100.000
python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 128 -j 1 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN --in-shape 3 224 224

python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 128 -j 1 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
--routing-name-list Tiny_FPN Tiny_FPN Tiny_FPN --in-shape 3 224 224
#
## * Acc@1 94.000 Acc@5 99.400
#python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 128 -j 8 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
#--routing-name-list FPN FPN --in-shape 3 224 224
#
#python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 512 -j 8 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half_gumbel \
#--routing-name-list FPN FPN FPN --in-shape 3 224 224

# ====================================== depth 1 ====================================== #
#python train_imagenet.py -d image_folder -a capsnet_em_depthx1 -b 64 -j 8 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_tiny_half --in-shape 3 224 224 \
#> log_capsnet_em_depthx1_sub_imagenet.txt
# * Acc@1 92.200 Acc@5 100.000

#python train_imagenet.py -d image_folder -a capsnet_sr_depthx1 -b 128 -j 8 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_tiny_half --in-shape 3 224 224 \
#> log_capsnet_sr_depthx1_sub_imagenet.txt
# * Acc@1 64.600 Acc@5 83.800

# ====================================== depth 2 ====================================== #
python train_imagenet.py -d image_folder -a capsnet_em_depthx2 -b 128 -j 8 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_tiny_half --in-shape 3 224 224 \
> log_capsnet_em_depthx2.txt
# * Acc@1 93.000 Acc@5 100.000
#
python train_imagenet.py -d image_folder -a capsnet_sr_depthx2 -b 128 -j 8 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_tiny_half --in-shape 3 224 224 \
> log_capsnet_sr_depthx2.txt

# ====================================== backbone ====================================== #
#python train_imagenet.py -d image_folder -a resnet18_tiny_half -b 128 -j 8 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --in-shape 3 224 224
echo "done."
