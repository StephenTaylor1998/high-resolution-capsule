from torchvision import datasets
from torchvision.transforms import transforms

cifar_normalize = transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                       std=[0.24703223, 0.24348513, 0.26158784])

cifar_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    cifar_normalize,
])

cifar_transform_test = transforms.Compose([
    # transforms.RandomRotation((45, 45)),
    transforms.ToTensor(),
    cifar_normalize,
])


def train_dataset(data_dir, transform=cifar_transform_train, **kwargs):
    return datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True, **kwargs)


# val dataset example for tiny-image-net
def val_dataset(data_dir, transform=cifar_transform_test, **kwargs):
    return datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True, **kwargs)


# test dataset example for tiny-image-net
def test_dataset(data_dir, transform=cifar_transform_test, **kwargs):
    return datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True, **kwargs)
