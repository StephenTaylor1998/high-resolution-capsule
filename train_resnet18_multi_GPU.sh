#python main.py -a resnet18 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/ILSVRC2012/imagenet2012/

#python train_imagenet.py -d imagefolder -a resnet18 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/ILSVRC2012/

#python train_imagenet.py -d cifar10 -a resnet18 -b 64 -j 8 -c 10 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset

#python train_imagenet.py -d cifar10 -a resnet50_tiny -b 32 -j 4 -c 10 ./data/dataset

#python train_imagenet.py -d fashion_mnist -a resnet50_tiny_c1 -b 128 -j 4 -c 10 ./data/dataset
#python train_imagenet.py -d fashion_mnist -a resnet50_tiny_c1 -b 128 -j 4 -c 10 ./data/dataset \
#--resume ./data/checkpoint.pth.tar --epoch 150

#python train_imagenet.py -d fashion_mnist -a resnet50_tiny_c1 -b 128 -j 4 -c 10 -e ./data/dataset \
#--resume ./data/checkpoint.pth.tar

#python train_imagenet.py -d cifar10 -a resnet50 -b 128 -j 8 -c 10 ./data/dataset --epoch 500 \
#--resume ./data/model_best.pth.tar
#--pretrained

#python train_imagenet.py -d fashion_mnist -a resnet50_tiny_c1 -b 128 -j 4 -c 10 -e ./data/dataset \
#--resume ./data/checkpoint.pth.tar


#python train_imagenet.py -d cifar10 -a densenet121 -b 640 -j 32 -c 10 --epoch 400 \
#--dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset \
#--resume ./data/checkpoint.pth.tar --weight-decay 5e-4

#python train_imagenet.py -d cifar10 -a densenet121 -b 128 -j 16 -c 10 --epoch 400 ./data/dataset \
#--resume ./data/checkpoint.pth.tar --weight-decay 5e-4


#python train_imagenet.py -d cifar10 -a resnet18_cifar -b 512 -j 32 -c 10 --epoch 400 --dist-url 'tcp://127.0.0.1:8889' \
#--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset --weight-decay 5e-4
#--resume ./data/checkpoint.pth.tar

#python train_imagenet.py -d cifar10 -a resnet18_cifar -b 512 -j 32 -c 10 --epoch 400 -e --dist-url 'tcp://127.0.0.1:8889' \
#--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset --weight-decay 5e-4 \
#--resume ./data/model_best.pth.tar
#--resume ./data/checkpoint.pth.tar

#python train_imagenet.py -d cifar10 -a resnet18_cifar -b 512 -j 32 -c 10 --epoch 400 --dist-url 'tcp://127.0.0.1:8889' \
#--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset --weight-decay 5e-4 \
#--resume ./data/model_best.pth.tar


#python train_imagenet.py -d cifar10 -a resnet50 -b 640 -j 32 -c 10 --epoch 400 \
#--dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset \
##--resume ./data/checkpoint.pth.tar --weight-decay 5e-4

#python train_imagenet.py -d mnist -a capsule_efficient_without_docoder -b 32 -j 32 -c 10 --epoch 100 --dist-url 'tcp://127.0.0.1:8889' \
#--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset --weight-decay 1e-4

#python train_imagenet.py -d fashion_mnist -a capsule_hinton_without_docoder -b 256 -j 32 -c 10 --epoch 100 \
#./data/dataset --weight-decay 1e-4 --lr-scheduler mnist --lr 0.001


python train_imagenet.py -d mnist -a capsule_efficient_without_docoder -b 256 -j 32 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' \
--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset --weight-decay 1e-7 --lr 0.001
