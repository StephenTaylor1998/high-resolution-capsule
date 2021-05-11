#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo "Training..."
# ====================================== multi gpu ====================================== #
python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN --in-shape 3 224 224

python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN FPN FPN --in-shape 3 224 224

python train_imagenet.py -d image_folder_high_resolution -a hr_caps_r_fpn -b 128 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN --in-shape 3 512 512

python train_imagenet.py -d image_folder_high_resolution -a hr_caps_r_fpn -b 128 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN FPN FPN --in-shape 3 512 512

python train_imagenet.py -d image_folder -a capsnet_em_r20_routingx1 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --in-shape 3 224 224

python train_imagenet.py -d image_folder -a resnet18_dwt_tiny_half -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar

python train_imagenet.py -d image_folder -a resnet18_tiny_half -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar

# ====================================== single gpu ====================================== #

echo "done."

echo "Testing..."

python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN --in-shape 3 224 224 --resume ./data/weights/hr_caps_r_fpn_epoch300_bs51_lr0.1_image_folder_FPN/model_best.pth.tar -t

python train_imagenet.py -d image_folder -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar --backbone resnet18_dwt_tiny_half \
--routing-name-list FPN FPN FPN --in-shape 3 224 224 --resume ./data/weights/hr_caps_r_fpn_epoch300_bs51_lr0.1_image_folder_FPN_FPN_FPN/model_best.pth.tar -t

python train_imagenet.py -d image_folder -a resnet18_dwt_tiny_half -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar \
 --resume ./data/weights/resnet18_dwt_tiny_half_epoch300_bs51_lr0.1_image_folder_FPN/model_best.pth.tar -t

python train_imagenet.py -d image_folder -a resnet18_tiny_half -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar \
 --resume ./data/weights/resnet50_dwt_tiny_half_epoch300_bs51_lr0.1_image_folder_FPN/model_best.pth.tar -t

echo "done."
