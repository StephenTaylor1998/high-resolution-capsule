from torchvision.transforms import transforms, functional

__all__ = ['ImageNetNormalize',
           'ImageNetTrainTransform',
           'ImageNetValidationTransform',
           'ImageNetTestTransform',
           'HRImageNetTrainTransform',
           'HRImageNetValidationTransform',
           'HRImageNetTestTransform',
           'MNISTTrainTransform',
           'MNISTValidationTransform',
           'MNISTTestTransform']

ImageNetNormalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

# iamgenet examples
ImageNetTrainTransform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ImageNetNormalize,
])

ImageNetValidationTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ImageNetNormalize,
])

ImageNetTestTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomRotation((165, 165)),
    transforms.ToTensor(),
    ImageNetNormalize,
])

# iamgenet examples
HRImageNetTrainTransform = transforms.Compose([
    transforms.RandomResizedCrop(512),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ImageNetNormalize,
])

HRImageNetValidationTransform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    ImageNetNormalize,
])

HRImageNetTestTransform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    # transforms.RandomRotation((105, 105)),
    transforms.ToTensor(),
    ImageNetNormalize,
])


# mnist
MNISTTrainTransform = transforms.Compose([
    transforms.RandomRotation(30, center=(14, 14)),
    # transforms.RandomPerspective(),
    transforms.RandomResizedCrop(28, (0.85, 1.15), (0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456], std=[0.225]),
])

MNISTValidationTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456], std=[0.225]),
])

MNISTTestTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456], std=[0.225]),
])
