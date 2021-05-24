import torch
from torchvision import datasets
from torchvision.transforms import transforms


cifar_normalize = transforms.Normalize(mean=[0.43090966, 0.4302428, 0.44634357],
                                       std=[0.19759192, 0.20029082, 0.19811132])

cifar_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=1),
    transforms.RandomRotation((-30, 30)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    cifar_normalize,
])

cifar_transform_test = transforms.Compose([
    # transforms.RandomRotation((45, 45)),
    transforms.ToTensor(),
    cifar_normalize,
])


def train_dataset(data_dir, transform=cifar_transform_train, split_size=10000, **kwargs):
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
