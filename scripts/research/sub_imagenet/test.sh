#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo "Training..."

# ====================================== hr caps ====================================== #
# * Acc@1 94.400 Acc@5 100.000
python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 512 -j 8 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN --in-shape 3 224 224 --resume

# * Acc@1 94.000 Acc@5 99.400
python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 512 -j 8 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN FPN --in-shape 3 224 224

python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 512 -j 8 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN FPN FPN --in-shape 3 224 224

# ====================================== depth 2 ====================================== #
python train_imagenet.py -d image_folder -a capsnet_em_depthx2 -b 512 -j 8 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half --in-shape 3 224 224

python train_imagenet.py -d image_folder -a capsnet_sr_depthx2 -b 512 -j 8 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half --in-shape 3 224 224

echo "done."
