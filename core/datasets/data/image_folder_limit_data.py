import os
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from core.datasets.transforms.custom_transform import *

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


# train dataset example for image-net
def train_dataset(data_dir, transform=ImageNetTrainTransform, split_rate=0.2):
    data_dir = os.path.join(data_dir, 'train')
    train_data = datasets.ImageFolder(data_dir, transform)
    length = len(train_data)
    print(length)
    train_size, validate_size = int(length * split_rate), length - int(length * split_rate)
    train_set, validate_set = torch.utils.data.random_split(
        dataset=train_data,
        lengths=[train_size, validate_size],
        generator=torch.Generator().manual_seed(42))
    return train_set


# val dataset example for image-net
def val_dataset(data_dir, transform=ImageNetValidationTransform):
    data_dir = os.path.join(data_dir, 'val')
    return datasets.ImageFolder(data_dir, transform)


# test dataset example for image-net
def test_dataset(data_dir, transform=ImageNetTestTransform):
    data_dir = os.path.join(data_dir, 'val')
    return datasets.ImageFolder(data_dir, transform)


# if __name__ == '__main__':
#     dataset = train_dataset("../../../data/dataset/sub_imagenet")
#     print(len(dataset))
#     from torch.utils import data
#     # data.random_split()
