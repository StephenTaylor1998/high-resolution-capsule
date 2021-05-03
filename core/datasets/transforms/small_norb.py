import torch
import numpy as np
from torchvision import transforms


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


class DefaultArgs(object):
    def __init__(self):
        super(DefaultArgs, self).__init__()
        self.crop_dim = None
        self.brightness = None
        self.contrast = None


def get_transform(args):
    return {
        'train_valid': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((args.crop_dim, args.crop_dim)),
            transforms.ColorJitter(brightness=args.brightness / 255., contrast=args.contrast),
            transforms.ToTensor(),
            Standardize()]),

        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((args.crop_dim, args.crop_dim)),
            transforms.ToTensor(),
            Standardize()]),
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((args.crop_dim, args.crop_dim)),
            transforms.ColorJitter(brightness=args.brightness / 255., contrast=args.contrast),
            transforms.ToTensor(),
            Standardize()]),
        'valid': transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((args.crop_dim, args.crop_dim)),
            transforms.ToTensor(),
            Standardize()])
    }
