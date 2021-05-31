import os
import argparse
import json
import warnings

from .train import ckpt_link


def get_args():
    """Parse arguments and set runtime defaults.
    """
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

    if args.mode == 'train':
        set_train_args(args)
    elif args.mode == 'test':
        set_test_args(args)

    return args


def add_common_args(parser):
    parser.add_argument('--in-norms', type=str_list, help='comma-sep. list '
            'of input normalization functions')
    parser.add_argument('--tgt-norms', type=str_list, help='comma-sep. list '
            'of target normalization functions')
    parser.add_argument('--crop', type=int_tuple,
            help='size to crop the input and target data. Default is the '
            'field size. Comma-sep. list of 1 or d integers')
    parser.add_argument('--crop-start', type=int_tuple,
            help='starting point of the first crop. Default is the origin. '
            'Comma-sep. list of 1 or d integers')
    parser.add_argument('--crop-stop', type=int_tuple,
            help='stopping point of the last crop. Default is the opposite '
            'corner to the origin. Comma-sep. list of 1 or d integers')
    parser.add_argument('--crop-step', type=int_tuple,
            help='spacing between crops. Default is the crop size. '
            'Comma-sep. list of 1 or d integers')
    parser.add_argument('--in-pad', '--pad', default=0, type=int_tuple,
            help='size to pad the input data beyond the crop size, assuming '
            'periodic boundary condition. Comma-sep. list of 1, d, or dx2 '
            'integers, to pad equally along all axes, symmetrically on each, '
            'or by the specified size on every boundary, respectively')
    parser.add_argument('--tgt-pad', default=0, type=int_tuple,
            help='size to pad the target data beyond the crop size, assuming '
            'periodic boundary condition, useful for super-resolution. '
            'Comma-sep. list with the same format as --in-pad')
    parser.add_argument('--scale-factor', default=1, type=int,
            help='upsampling factor for super-resolution, in which case '
            'crop and pad are sizes of the input resolution')

    parser.add_argument('--model', type=str, required=True,
            help='(generator) model')
    parser.add_argument('--criterion', default='MSELoss', type=str,
            help='loss function')
    parser.add_argument('--load-state', default=ckpt_link, type=str,
            help='path to load the states of model, optimizer, rng, etc. '
            'Default is the checkpoint. '
            'Start from scratch in case of empty string or missing checkpoint')
    parser.add_argument('--load-state-non-strict', action='store_false',
            help='allow incompatible keys when loading model states',
            dest='load_state_strict')

    # somehow I named it "batches" instead of batch_size at first
    # "batches" is kept for now for backward compatibility
    parser.add_argument('--batch-size', '--batches', type=int, required=True,
            help='mini-batch size, per GPU in training or in total in testing')
    parser.add_argument('--loader-workers', default=8, type=int,
            help='number of subprocesses per data loader. '
            '0 to disable multiprocessing')

    parser.add_argument('--callback-at', type=lambda s: os.path.abspath(s),
            help='directory of custorm code defining callbacks for models, '
            'norms, criteria, and optimizers. Disabled if not set. '
            'This is appended to the default locations, '
            'thus has the lowest priority')
    parser.add_argument('--misc-kwargs', default='{}', type=json.loads,
            help='miscellaneous keyword arguments for custom models and '
            'norms. Be careful with name collisions')


def add_train_args(parser):
    add_common_args(parser)

    parser.add_argument('--train-style-pattern', type=str, required=True,
            help='glob pattern for training data styles')
    parser.add_argument('--train-in-patterns', type=str_list, required=True,
            help='comma-sep. list of glob patterns for training input data')
    parser.add_argument('--train-tgt-patterns', type=str_list, required=True,
            help='comma-sep. list of glob patterns for training target data')
    parser.add_argument('--val-style-pattern', type=str,
            help='glob pattern for validation data styles')
    parser.add_argument('--val-in-patterns', type=str_list,
            help='comma-sep. list of glob patterns for validation input data')
    parser.add_argument('--val-tgt-patterns', type=str_list,
            help='comma-sep. list of glob patterns for validation target data')
    parser.add_argument('--augment', action='store_true',
            help='enable data augmentation of axis flipping and permutation')
    parser.add_argument('--aug-shift', type=int_tuple,
            help='data augmentation by shifting cropping by [0, aug_shift) pixels, '
            'useful for models that treat neighboring pixels differently, '
            'e.g. with strided convolutions. '
            'Comma-sep. list of 1 or d integers')
    parser.add_argument('--aug-add', type=float,
            help='additive data augmentation, (normal) std, '
            'same factor for all fields')
    parser.add_argument('--aug-mul', type=float,
            help='multiplicative data augmentation, (log-normal) std, '
            'same factor for all fields')

    parser.add_argument('--optimizer', default='Adam', type=str,
            help='optimization algorithm')
    parser.add_argument('--lr', type=float, required=True,
            help='initial learning rate')
    parser.add_argument('--optimizer-args', default='{}', type=json.loads,
            help='optimizer arguments in addition to the learning rate, '
            'e.g. --optimizer-args \'{"betas": [0.5, 0.9]}\'')
    parser.add_argument('--reduce-lr-on-plateau', action='store_true',
            help='Enable ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--scheduler-args', default='{"verbose": true}',
            type=json.loads,
            help='arguments for the ReduceLROnPlateau scheduler')
    parser.add_argument('--init-weight-std', type=float,
            help='weight initialization std')
    parser.add_argument('--epochs', default=128, type=int,
            help='total number of epochs to run')
    parser.add_argument('--seed', default=42, type=int,
            help='seed for initializing training')

    parser.add_argument('--div-data', action='store_true',
            help='enable data division among GPUs for better page caching. '
            'Data division is shuffled every epoch. '
            'Only relevant if there are multiple crops in each field')
    parser.add_argument('--div-shuffle-dist', default=1, type=float,
            help='distance to further shuffle cropped samples relative to '
            'their fields, to be used with --div-data. '
            'Only relevant if there are multiple crops in each file. '
            'The order of each sample is randomly displaced by this value. '
            'Setting it to 0 turn off this randomization, and setting it to N '
            'limits the shuffling within a distance of N files. '
            'Change this to balance cache locality and stochasticity')
    parser.add_argument('--dist-backend', default='nccl', type=str,
            choices=['gloo', 'nccl'], help='distributed backend')
    parser.add_argument('--log-interval', default=100, type=int,
            help='interval (batches) between logging training loss')
    parser.add_argument('--detect-anomaly', action='store_true',
            help='enable anomaly detection for the autograd engine')


def add_test_args(parser):
    add_common_args(parser)

    parser.add_argument('--test-style-pattern', type=str, required=True,
            help='glob pattern for test data styles')
    parser.add_argument('--test-in-patterns', type=str_list, required=True,
            help='comma-sep. list of glob patterns for test input data')
    parser.add_argument('--test-tgt-patterns', type=str_list, required=True,
            help='comma-sep. list of glob patterns for test target data')

    parser.add_argument('--num-threads', type=int,
            help='number of CPU threads when cuda is unavailable. '
            'Default is the number of CPUs on the node by slurm')


def str_list(s):
    return s.split(',')


def int_tuple(s):
    t = s.split(',')
    t = tuple(int(i) for i in t)
    if len(t) == 1:
        return t[0]
    else:
        return t


def set_common_args(args):
    pass


def set_train_args(args):
    set_common_args(args)

    args.val = args.val_in_patterns is not None and \
            args.val_tgt_patterns is not None


def set_test_args(args):
    set_common_args(args)
