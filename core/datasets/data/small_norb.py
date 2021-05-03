import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from core.datasets.transforms.small_norb import get_transform, random_split


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


def smallnorb(args, dataset_paths):
    transf = get_transform(args)
    config = {'train_valid': True, 'test': False}
    datasets = {i: smallNORB(dataset_paths[i], transform=transf[i],
                             shuffle=config[i]) for i in config.keys()}

    data, labels = random_split(data=datasets['train_valid'].data,
                                labels=datasets['train_valid'].labels,
                                n_classes=5,
                                # % of train set per class
                                n_samples_per_class=np.unique(
                                    datasets['train_valid'].labels, return_counts=True)[1] // 5)

    # make new training set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
                                      labels=labels['train'], transform=transf['train'])

    # make class balanced validation set
    datasets['valid'] = CustomDataset(data=data['valid'],
                                      labels=labels['valid'], transform=transf['valid'])

    config = {'train': True, 'train_valid': True,
              'valid': False, 'test': False}

    dataloaders = {i: DataLoader(datasets[i], shuffle=config[i], pin_memory=True,
                                 num_workers=8, batch_size=args.batch_size) for i in config.keys()}

    return dataloaders


def train_valid_dataset(data_dir, args, **kwargs):
    transf = get_transform(args)['train_valid']
    return smallNORB(data_dir, transform=transf['train_valid'], shuffle=True)


def test_dataset(data_dir, args, **kwargs):
    transf = get_transform(args)['test']
    return smallNORB(data_dir, transform=transf['test'], shuffle=False)


def train_dataset(data_dir, args, **kwargs):
    transf = get_transform(args)['train_valid']
    train_valid = smallNORB(data_dir, transform=transf['train_valid'], shuffle=True)
    data, labels = random_split(data=train_valid['train_valid'].data,
                                labels=train_valid['train_valid'].labels,
                                n_classes=5,
                                n_samples_per_class=np.unique(
                                    train_valid['train_valid'].labels, return_counts=True
                                )[1] // 5)

    return CustomDataset(data=data['train'], labels=labels['train'], transform=transf['train'])


def valid_dataset(data_dir, args, **kwargs):
    transf = get_transform(args)['train_valid']
    train_valid = smallNORB(data_dir, transform=transf['train_valid'], shuffle=False)
    data, labels = random_split(data=train_valid['train_valid'].data,
                                labels=train_valid['train_valid'].labels,
                                n_classes=5,
                                n_samples_per_class=np.unique(
                                    train_valid['train_valid'].labels, return_counts=True
                                )[1] // 5)

    return CustomDataset(data=data['valid'], labels=labels['valid'], transform=transf['valid'])
