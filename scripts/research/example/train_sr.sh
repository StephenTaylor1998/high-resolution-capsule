


python train_imagenet.py -d image_folder -a capsnet_sr_routingx1 -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar

python train_imagenet.py -d cifar10 -a capsnet_sr_routingx1 -b 256 -j 20 -c 10 --epoch 300 ./data/dataset/ --lr-scheduler cifar


python train_imagenet.py -d image_folder -a capsnet_sr_routingx1 -b 256 -j 20 -c 10 --epoch 300 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/sub_imagenet --lr-scheduler cifar

