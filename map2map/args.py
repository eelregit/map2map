import os
import argparse
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
            'of input normalization functions from .data.norms')
    parser.add_argument('--tgt-norms', type=str_list, help='comma-sep. list '
            'of target normalization functions from .data.norms')
    parser.add_argument('--crop', type=int,
            help='size to crop the input and target data')
    parser.add_argument('--pad', default=0, type=int,
            help='size to pad the input data beyond the crop size, assuming '
            'periodic boundary condition')
    parser.add_argument('--scale-factor', default=1, type=int,
            help='input upsampling factor for super-resolution purpose, in '
            'which case crop and pad will be taken at the original resolution')

    parser.add_argument('--model', required=True, type=str,
            help='model from .models')
    parser.add_argument('--criterion', default='MSELoss', type=str,
            help='model criterion from torch.nn')
    parser.add_argument('--load-state', default=ckpt_link, type=str,
            help='path to load the states of model, optimizer, rng, etc. '
            'Default is the checkpoint. '
            'Start from scratch if set empty or the checkpoint is missing')
    parser.add_argument('--load-state-non-strict', action='store_false',
            help='allow incompatible keys when loading model states',
            dest='load_state_strict')

    parser.add_argument('--batches', default=1, type=int,
            help='mini-batch size, per GPU in training or in total in testing')
    parser.add_argument('--loader-workers', type=int,
            help='number of data loading workers, per GPU in training or '
            'in total in testing. '
            'Default is the batch size or 0 for batch size 1')

    parser.add_argument('--cache', action='store_true',
            help='enable LRU cache of input and target fields to reduce I/O')
    parser.add_argument('--cache-maxsize', type=int,
            help='maximum pairs of fields in cache, unlimited by default. '
            'This only applies to training if not None, '
            'in which case the testing cache maxsize is 1')
    parser.add_argument('--callback-at', type=lambda s: os.path.abspath(s),
            help='directory of custorm code defining callbacks for models, '
            'norms, criteria, and optimizers. Disabled if not set. '
            'This is appended to the default locations, '
            'thus has the lowest priority.')


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
            help='enable data augmentation of axis flipping and permutation')
    parser.add_argument('--aug-add', type=float,
            help='additive data augmentation, (normal) std, '
            'same factor for all fields')
    parser.add_argument('--aug-mul', type=float,
            help='multiplicative data augmentation, (log-normal) std, '
            'same factor for all fields')

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
    parser.add_argument('--output', type=str, required=True,
            help='output file name')
    parser.add_argument('--use-openvino', action='store_true',
            help='use OpenVINO for speed in inference. You must provide additional arguments')
    parser.add_argument('--openvino-pre-model', action='store_true',
            help='please provide OpenVINO files. This will prevent trace de code again and save you time')
    parser.add_argument('--openvino-xml-file', type=str, default='model.xml',
            help='please provide OpenVINO xml file')
    parser.add_argument('--openvino-bin-file', type=str, default='model.bin',
            help='please provide OpenVINO bin file')        
    parser.add_argument('--onnx-file', type=str, default='model.onnx',
            help='please provide ONNX file name to be saved')

    
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


def set_common_args(args):
    if args.loader_workers is None:
        args.loader_workers = 0
        if args.batches > 1:
            args.loader_workers = args.batches

    if not args.cache and args.cache_maxsize is not None:
        args.cache_maxsize = None
        warnings.warn('Resetting cache maxsize given cache is disabled',
                      RuntimeWarning)
    if (args.cache and args.cache_maxsize is not None
            and args.cache_maxsize < 1):
        args.cache = False
        args.cache_maxsize = None
        warnings.warn('Disabling cache given cache maxsize < 1',
                      RuntimeWarning)


def set_train_args(args):
    set_common_args(args)

    args.val = args.val_in_patterns is not None and \
            args.val_tgt_patterns is not None

    args.adv = args.adv_model is not None

    if args.adv:
        if args.adv_lr is None:
            args.adv_lr = args.lr
        if args.adv_weight_decay is None:
            args.adv_weight_decay = args.weight_decay


def set_test_args(args):
    set_common_args(args)
