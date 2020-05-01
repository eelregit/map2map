import argparse

from .train import ckpt_link


def get_args():
    parser = argparse.ArgumentParser(
        description='Transform field(s) to field(s)')
    subparsers = parser.add_subparsers(title='modes', dest='mode', required=True)
    train_parser = subparsers.add_parser(
        'train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    test_parser = subparsers.add_parser(
        'test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_train_args(train_parser)
    add_test_args(test_parser)

    args = parser.parse_args()

    return args


def add_common_args(parser):
    parser.add_argument('--in-norms', type=str_list, help='comma-sep. list '
            'of input normalization functions from .data.norms')
    parser.add_argument('--tgt-norms', type=str_list, help='comma-sep. list '
            'of target normalization functions from .data.norms')
    parser.add_argument('--crop', type=int,
            help='size to crop the input and target data')
    parser.add_argument('--pad', default=0, type=int,
            help='size to pad the input data beyond the crop size, assuming '
            'periodic boundary condition')
    parser.add_argument('--scale-factor', default=1, type=int,
            help='upsampling factor for super-resolution, in which case '
            'crop and pad are sizes of the input resolution')

    parser.add_argument('--model', required=True, type=str,
            help='model from .models')
    parser.add_argument('--criterion', default='MSELoss', type=str,
            help='model criterion from torch.nn')
    parser.add_argument('--load-state', default=ckpt_link, type=str,
            help='path to load the states of model, optimizer, rng, etc. '
            'Default is the checkpoint. '
            'Start from scratch if the checkpoint does not exist')
    parser.add_argument('--load-state-non-strict', action='store_false',
            help='allow incompatible keys when loading model states',
            dest='load_state_strict')

    parser.add_argument('--batches', default=1, type=int,
            help='mini-batch size, per GPU in training or in total in testing')
    parser.add_argument('--loader-workers', default=0, type=int,
            help='number of data loading workers, per GPU in training or '
            'in total in testing')

    parser.add_argument('--cache', action='store_true',
            help='enable caching in field datasets')


def add_train_args(parser):
    add_common_args(parser)

    parser.add_argument('--train-in-patterns', type=str_list, required=True,
            help='comma-sep. list of glob patterns for training input data')
    parser.add_argument('--train-tgt-patterns', type=str_list, required=True,
            help='comma-sep. list of glob patterns for training target data')
    parser.add_argument('--val-in-patterns', type=str_list,
            help='comma-sep. list of glob patterns for validation input data')
    parser.add_argument('--val-tgt-patterns', type=str_list,
            help='comma-sep. list of glob patterns for validation target data')
    parser.add_argument('--augment', action='store_true',
            help='enable training data augmentation')

    parser.add_argument('--adv-model', type=str,
            help='enable adversary model from .models')
    parser.add_argument('--adv-model-spectral-norm', action='store_true',
            help='enable spectral normalization on the adversary model')
    parser.add_argument('--adv-criterion', default='BCEWithLogitsLoss', type=str,
            help='adversarial criterion from torch.nn')
    parser.add_argument('--min-reduction', action='store_true',
            help='enable minimum reduction in adversarial criterion')
    parser.add_argument('--cgan', action='store_true',
            help='enable conditional GAN')
    parser.add_argument('--adv-start', default=0, type=int,
            help='epoch to start adversarial training')
    parser.add_argument('--adv-label-smoothing', default=1, type=float,
            help='label of real samples for the adversary model, '
            'e.g. 0.9 for label smoothing and 1 to disable')
    parser.add_argument('--loss-fraction', default=0.5, type=float,
            help='final fraction of loss (vs adv-loss)')
    parser.add_argument('--loss-halflife', default=20, type=float,
            help='half-life (epoch) to anneal loss while enhancing adv-loss')
    parser.add_argument('--instance-noise', default=0, type=float,
            help='noise added to the adversary inputs to stabilize training')
    parser.add_argument('--instance-noise-batches', default=1e4, type=float,
            help='noise annealing duration')

    parser.add_argument('--optimizer', default='Adam', type=str,
            help='optimizer from torch.optim')
    parser.add_argument('--lr', default=0.001, type=float,
            help='initial learning rate')
#    parser.add_argument('--momentum', default=0.9, type=float,
#            help='momentum')
    parser.add_argument('--weight-decay', default=0, type=float,
            help='weight decay')
    parser.add_argument('--adv-lr', type=float,
            help='initial adversary learning rate')
    parser.add_argument('--adv-weight-decay', type=float,
            help='adversary weight decay')
    parser.add_argument('--reduce-lr-on-plateau', action='store_true',
            help='Enable ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--init-weight-std', type=float,
            help='weight initialization std')
    parser.add_argument('--epochs', default=128, type=int,
            help='total number of epochs to run')
    parser.add_argument('--seed', default=42, type=int,
            help='seed for initializing training')

    parser.add_argument('--div-data', action='store_true',
            help='enable data division among GPUs, useful with cache')
    parser.add_argument('--dist-backend', default='nccl', type=str,
            choices=['gloo', 'nccl'], help='distributed backend')
    parser.add_argument('--log-interval', default=100, type=int,
            help='interval between logging training loss')


def add_test_args(parser):
    add_common_args(parser)

    parser.add_argument('--test-in-patterns', type=str_list, required=True,
            help='comma-sep. list of glob patterns for test input data')
    parser.add_argument('--test-tgt-patterns', type=str_list, required=True,
            help='comma-sep. list of glob patterns for test target data')


def str_list(s):
    return s.split(',')


#def int_tuple(t):
#    t = t.split(',')
#    t = tuple(int(i) for i in t)
#    if len(t) == 1:
#        t = t[0]
#    elif len(t) != 6:
#        raise ValueError('size must be int or 6-tuple')
#    return t
