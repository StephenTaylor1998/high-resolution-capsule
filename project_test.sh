# train
#python test_project.py -a resnet18 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --epoch 1 --multiprocessing-distributed --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/ILSVRC2012/

# eval
# python test_project.py -a resnet18 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --epoch 1 --multiprocessing-distributed --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/ILSVRC2012/ -e


python test_project.py /home/aistudio/Desktop/datasets/ILSVRC2012/ -e --gpu 0 -b 1