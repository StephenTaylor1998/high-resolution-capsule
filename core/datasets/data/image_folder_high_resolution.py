import os
from torchvision import datasets
from torchvision.transforms import transforms
from core.datasets.transforms.custom_transform import *


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def classify_dataset(data_dir, transform, not_strict=False):
    if not os.path.exists(data_dir) and not_strict:
        print("path ==> '%s' is not found" % data_dir)
        return

    return datasets.ImageFolder(data_dir, transform)


# train dataset example for image-net
def train_dataset(data_dir, transform=HRImageNetTrainTransform):
    data_dir = os.path.join(data_dir, 'train')
    return datasets.ImageFolder(data_dir, transform)


# val dataset example for image-net
def val_dataset(data_dir, transform=HRImageNetValidationTransform):
    data_dir = os.path.join(data_dir, 'val')
    return datasets.ImageFolder(data_dir, transform)


# test dataset example for image-net
def test_dataset(data_dir, transform=HRImageNetTestTransform):
    data_dir = os.path.join(data_dir, 'val')
    return datasets.ImageFolder(data_dir, transform)
