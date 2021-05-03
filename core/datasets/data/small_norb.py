import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    """ Creates a custom pytorch dataset, mainly
        used for creating validation set splits. """

    def __init__(self, data, labels, transform=None):
        # shuffle the dataset
        idx = np.random.permutation(data.shape[0])
        if isinstance(data, torch.Tensor):
            data = data.numpy()  # to work with `ToPILImage'
        self.data = data[idx]
        self.labels = labels[idx]
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.data[idx])
        else:
            print("with out transform")
            image = self.data[idx]
        return image, self.labels[idx]


def random_split(data, labels, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set. """
    train_X, train_Y, valid_X, valid_Y = [], [], [], []

    for c in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == c).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[c], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_X.extend(data[train_samples])
        train_Y.extend(labels[train_samples])
        valid_X.extend(data[valid_samples])
        valid_Y.extend(labels[valid_samples])

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'train': torch.stack(train_X), 'valid': torch.stack(valid_X)}, \
               {'train': torch.stack(train_Y), 'valid': torch.stack(valid_Y)}
    else:
        # transforms list of np arrays to tensor
        return {'train': torch.from_numpy(np.stack(train_X)), 'valid': torch.from_numpy(np.stack(valid_X))}, \
               {'train': torch.from_numpy(np.stack(train_Y)), 'valid': torch.from_numpy(np.stack(valid_Y))}


class Standardize(object):
    """ Standardizes a 'PIL Image' such that each channel
        gets zero mean and unit variance. """

    def __call__(self, img):
        return (img - img.mean(dim=(1, 2), keepdim=True)) / torch.clamp(img.std(dim=(1, 2), keepdim=True), min=1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def sample_weights(labels):
    """ Calculates per sample weights. """
    class_sample_count = np.unique(labels, return_counts=True)[1]
    class_weights = 1. / torch.Tensor(class_sample_count)
    return class_weights[list(map(int, labels))]


def smallnorb(args, dataset_paths):
    transf = {'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop((args.crop_dim, args.crop_dim)),
        transforms.ColorJitter(brightness=args.brightness / 255., contrast=args.contrast),
        transforms.ToTensor(),
        Standardize()]),
        # transforms.Normalize((0.75239172, 0.75738262), (0.1758033 , 0.17200065))]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((args.crop_dim, args.crop_dim)),
            transforms.ToTensor(),
            Standardize()])}
    # transforms.Normalize((0.75239172, 0.75738262), (0.1758033 , 0.17200065))])}

    config = {'train': True, 'test': False}
    datasets = {i: smallNORB(dataset_paths[i], transform=transf[i],
                             shuffle=config[i]) for i in config.keys()}

    # return data, labels dicts for new train set and class-balanced valid set
    data, labels = random_split(data=datasets['train'].data,
                                labels=datasets['train'].labels,
                                n_classes=5,
                                n_samples_per_class=np.unique(
                                    datasets['train'].labels, return_counts=True)[1] // 5)  # % of train set per class

    # define transforms for train set (without valid data)
    transf['train_'] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop((args.crop_dim, args.crop_dim)),
        transforms.ColorJitter(brightness=args.brightness / 255., contrast=args.contrast),
        transforms.ToTensor(),
        Standardize()])
    # transforms.Normalize((0.75239172, 0.75738262), (0.1758033 , 0.17200065))])

    # define transforms for class-balanced valid set
    transf['valid'] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((args.crop_dim, args.crop_dim)),
        transforms.ToTensor(),
        Standardize()])
    # transforms.Normalize((0.75239172, 0.75738262), (0.1758033 , 0.17200065))])

    # save original full training set
    datasets['train_valid'] = datasets['train']

    # make new training set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
                                      labels=labels['train'], transform=transf['train_'])

    # make class balanced validation set
    datasets['valid'] = CustomDataset(data=data['valid'],
                                      labels=labels['valid'], transform=transf['valid'])

    config = {'train': True, 'train_valid': True,
              'valid': False, 'test': False}

    dataloaders = {i: DataLoader(datasets[i], shuffle=config[i], pin_memory=True,
                                 num_workers=8, batch_size=args.batch_size) for i in config.keys()}

    return dataloaders


class smallNORB(Dataset):
    """ In:
            data_path (string): path to the dataset split folder, i.e. train/valid/test
            transform (callable, optional): transform to be applied on a sample.
        Out:
            sample (dict): sample data and respective label"""

    def __init__(self, data_path, shuffle=True, transform=None):

        self.data_path = data_path
        self.shuffle = shuffle
        self.transform = transform
        self.data, self.labels = [], []

        # get path for each class folder
        for class_label_idx, class_name in enumerate(os.listdir(data_path)):
            class_path = os.path.join(data_path, class_name)

            # get name of each file per class and respective class name/label index
            for _, file_name in enumerate(os.listdir(class_path)):
                img = np.load(os.path.join(data_path, class_name, file_name))
                # Out ← [H, W, C] ← [C, H, W]
                if img.shape[0] < img.shape[1]:
                    img = np.moveaxis(img, 0, -1)
                self.data.extend([img])
                self.labels.append(class_label_idx)

        self.data = np.array(self.data, dtype=np.uint8)
        self.labels = np.array(self.labels)

        if self.shuffle:
            # shuffle the dataset
            idx = np.random.permutation(self.data.shape[0])
            self.data = self.data[idx]
            self.labels = self.labels[idx]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.data[idx])
        else:
            print("with out transform")
            image = self.data[idx]

        return image, self.labels[idx]  # (X, Y)
