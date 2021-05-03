import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from core.datasets.data.image_folder import classify_dataset
# from core.engine.classify import main_worker
from core.engine.distributed_engine import main_worker
from core.models.efficientnet import b7
from core.utils.argparse import arg_parse

parser = arg_parse()


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    model = b7()
    model.cuda()

    train_dataset = classify_dataset(os.path.join(args.data, 'train'))
    val_dataset = classify_dataset(os.path.join(args.data, 'val'))
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, model, train_dataset, val_dataset, args))
    else:
        main_worker(args.gpu, ngpus_per_node, model, train_dataset, val_dataset, args)


if __name__ == '__main__':
    main()
