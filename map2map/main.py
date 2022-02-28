from .args import get_args
from . import train
from . import test
from . import generate


def main():

    args = get_args()

    if args.mode == 'train':
        train.node_worker(args)
    elif args.mode == 'test':
        test.test(args)
    elif args.mode == 'generate':
        generate.generate(args)

if __name__ == '__main__':
    main()
