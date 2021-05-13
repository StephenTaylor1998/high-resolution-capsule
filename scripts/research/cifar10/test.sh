#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo "Training..."

# ====================================== hr caps ====================================== #
# * Acc@1 87.460 Acc@5 98.980 // FPN * 1
python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half \
--routing-name-list FPN --in-shape 3 32 32 --resume ./data/weights/hr_caps_r_fpn_epoch250_bs51_lr0.1_cifar10_FPNresnet10_tiny_half/checkpoint_epoch251.pth.tar -t

# * Acc@1 87.260 Acc@5 98.990 // FPN * 2
python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half \
--routing-name-list FPN FPN --in-shape 3 32 32 --resume ./data/weights/hr_caps_r_fpn_epoch250_bs51_lr0.1_cifar10_FPN_FPNresnet10_tiny_half/checkpoint_epoch251.pth.tar -t

# * Acc@1 87.280 Acc@5 98.830 // FPN * 3
python train_imagenet.py -d cifar10 -a hr_caps_r_fpn -b 512 -j 20 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half \
--routing-name-list FPN FPN FPN --in-shape 3 32 32 --resume ./data/weights/hr_caps_r_fpn_epoch250_bs51_lr0.1_cifar10_FPN_FPN_FPNresnet10_tiny_half/checkpoint_epoch251.pth.tar -t

# ====================================== depth 1 ====================================== #
# * Acc@1 85.810 Acc@5 98.710 // DR * 1  (Acc@1 86.410)
python train_imagenet.py -d cifar10 -a capsnet_dr_depthx1 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
--resume ./data/weights/capsnet_dr_depthx1_epoch300_bs51_lr0.1_cifar10resnet10_tiny_half/checkpoint_epoch301.pth.tar -t

# * Acc@1 85.710 Acc@5 98.560 // EM * 1  (Acc@1 86.050)
python train_imagenet.py -d cifar10 -a capsnet_em_depthx1 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
--resume ./data/weights/capsnet_em_depthx1_epoch300_bs51_lr0.1_cifar10resnet10_tiny_half/checkpoint_epoch301.pth.tar -t

# * Acc@1 84.270 Acc@5 98.470 // SR * 1  (Acc@1 84.420)
python train_imagenet.py -d cifar10 -a capsnet_sr_depthx1 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
--resume ./data/weights/capsnet_sr_depthx1_epoch300_bs51_lr0.1_cifar10resnet10_tiny_half/checkpoint_epoch301.pth.tar -t

# ====================================== depth 2 ====================================== #
# * Acc@1 86.900 Acc@5 98.480 // DR * 2  (Acc@1 86.790)
python train_imagenet.py -d cifar10 -a capsnet_dr_depthx2 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
--resume ./data/weights/capsnet_dr_depthx2_epoch300_bs51_lr0.1_cifar10resnet10_tiny_half/checkpoint_epoch301.pth.tar -t

# * Acc@1 68.850 Acc@5 96.460 // EM * 2  (Acc@1 63.370)
python train_imagenet.py -d cifar10 -a capsnet_em_depthx2 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
--resume ./data/weights/capsnet_em_depthx2_epoch300_bs51_lr0.1_cifar10resnet10_tiny_half/checkpoint_epoch301.pth.tar -t

# * Acc@1 45.840 Acc@5 91.240 // SR * 2  (Acc@1 44.860)
python train_imagenet.py -d cifar10 -a capsnet_sr_depthx2 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
--resume ./data/weights/capsnet_sr_depthx2_epoch300_bs51_lr0.1_cifar10resnet10_tiny_half/checkpoint_epoch301.pth.tar -t

# ====================================== depth 3 ====================================== #
# * Acc@1 10.420 Acc@5 50.480 // DR * 3  (Acc@1 10.350)
python train_imagenet.py -d cifar10 -a capsnet_dr_depthx3 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
--resume ./data/weights/capsnet_dr_depthx3_epoch300_bs51_lr0.1_cifar10resnet10_tiny_half/checkpoint_epoch301.pth.tar -t

# * Acc@1 38.170 Acc@5 87.370 // EM * 3  (Acc@1 37.610)
python train_imagenet.py -d cifar10 -a capsnet_em_depthx3 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
--resume ./data/weights/capsnet_em_depthx3_epoch300_bs51_lr0.1_cifar10resnet10_tiny_half/checkpoint_epoch301.pth.tar -t

# * Acc@1 37.020 Acc@5 88.840 // SR * 3  (Acc@1 35.180)
python train_imagenet.py -d cifar10 -a capsnet_sr_depthx3 -b 512 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_tiny_half --in-shape 3 32 32 \
--resume ./data/weights/capsnet_sr_depthx3_epoch300_bs51_lr0.1_cifar10resnet10_tiny_half/checkpoint_epoch301.pth.tar -t

echo "done."