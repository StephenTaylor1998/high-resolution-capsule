import torch
from torchvision import datasets
from torchvision.transforms import transforms


cifar_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                       std=[0.2023, 0.1994, 0.2010])

cifar_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=1),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    cifar_normalize,
])

cifar_transform_test = transforms.Compose([
    # transforms.RandomRotation((45, 45)),
    transforms.ToTensor(),
    cifar_normalize,
])


def train_dataset(data_dir, transform=cifar_transform_train, split_size=20000, **kwargs):
    train_data = datasets.SVHN(root=data_dir, split='train', transform=transform, download=True, **kwargs)
    length = len(train_data)
    print(length)
    train_size, validate_size = split_size, length - split_size
    train_set, _ = torch.utils.data.random_split(
        dataset=train_data,
        lengths=[train_size, validate_size],
        generator=torch.Generator().manual_seed(42))
    return train_set


# val dataset example for tiny-image-net
def val_dataset(data_dir, transform=cifar_transform_test, **kwargs):
    return datasets.SVHN(root=data_dir, split='test', transform=transform, download=True, **kwargs)


# test dataset example for tiny-image-net
def test_dataset(data_dir, transform=cifar_transform_test, **kwargs):
    return datasets.SVHN(root=data_dir, split='test', transform=transform, download=True, **kwargs)


'''
python train_imagenet.py -d svhn -a hr_caps_r_fpn -b 512 -j 2 -c 10 --epoch 250 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --lr-scheduler cifar --backbone resnet10_dwt_tiny_half --routing-name-list FPN FPN --in-shape 1 32 32
python train_imagenet.py -d svhn -a hr_caps_r_fpn -b 512 -j 2 -c 10 --epoch 250 ./data/dataset/svhn --lr-scheduler cifar --backbone resnet10_dwt_tiny_half --routing-name-list FPN FPN --in-shape 1 32 32
'''