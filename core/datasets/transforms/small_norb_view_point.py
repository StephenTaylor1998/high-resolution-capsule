from torchvision import transforms

SmallNorbVPTrainTransform = transforms.Compose([
    transforms.Resize(48),
    transforms.RandomCrop(32),
    transforms.ColorJitter(brightness=32. / 255, contrast=0.3),
    transforms.ToTensor(),
])

SmallNorbVPValTransform = transforms.Compose([
    transforms.Resize(48),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
])

SmallNorbVPTestTransform = transforms.Compose([
    transforms.Resize(48),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
])
