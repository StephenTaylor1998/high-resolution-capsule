# __all__ = {"imagenet", "cifar", "mnist", "get_scheduler_by_name"}


def imagenet(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cifar(optimizer, epoch, args):
    if epoch < 10:
        lr = args.lr * (epoch/10)
    elif epoch < 110:
        lr = args.lr
    elif epoch < 200:
        lr = args.lr * 0.1
    elif epoch < 250:
        lr = args.lr * 0.01
    else:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def mnist(optimizer, epoch, args):
    if epoch < 30:
        lr = args.lr
    elif epoch < 60:
        lr = args.lr * 0.5
    elif epoch < 90:
        lr = args.lr * 0.1
    elif epoch < 120:
        lr = args.lr * 0.05
    elif epoch < 150:
        lr = args.lr * 0.01
    else:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


_scheduler_dict = {"imagenet": imagenet,
                   "cifar": cifar,
                   "mnist": mnist,
                   }


def get_scheduler_by_name(name: str):
    return _scheduler_dict[name]
