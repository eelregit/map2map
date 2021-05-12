import sys
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FieldDataset
from .data import norms
from . import models
from .models import narrow_cast
from .utils import import_attr, load_model_state_dict


def test(args):
    print('pytorch {}'.format(torch.__version__))
    pprint(vars(args))
    sys.stdout.flush()

    test_dataset = FieldDataset(
        in_patterns=args.test_in_patterns,
        tgt_patterns=args.test_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=False,
        aug_shift=None,
        aug_add=None,
        aug_mul=None,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
    )

    in_chan, out_chan = test_dataset.in_chan, test_dataset.tgt_chan

    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(sum(in_chan), sum(out_chan), scale_factor=args.scale_factor,
                  **args.misc_kwargs)
    criterion = import_attr(args.criterion, torch.nn, callback_at=args.callback_at)
    criterion = criterion()

    device = torch.device('cpu')
    state = torch.load(args.load_state, map_location=device)
    load_model_state_dict(model, state['model'], strict=args.load_state_strict)
    print('model state at epoch {} loaded from {}'.format(
        state['epoch'], args.load_state))
    del state

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input, target = data['input'], data['target']

            output = model(input)
            input, output, target = narrow_cast(input, output, target)

            loss = criterion(output, target)

            print('sample {} loss: {}'.format(i, loss.item()))

            #if args.in_norms is not None:
            #    start = 0
            #    for norm, stop in zip(test_dataset.in_norms, np.cumsum(in_chan)):
            #        norm = import_attr(norm, norms, callback_at=args.callback_at)
            #        norm(input[:, start:stop], undo=True, **args.misc_kwargs)
            #        start = stop
            if args.tgt_norms is not None:
                start = 0
                for norm, stop in zip(test_dataset.tgt_norms, np.cumsum(out_chan)):
                    norm = import_attr(norm, norms, callback_at=args.callback_at)
                    norm(output[:, start:stop], undo=True, **args.misc_kwargs)
                    #norm(target[:, start:stop], undo=True, **args.misc_kwargs)
                    start = stop

            #test_dataset.assemble('_in', in_chan, input,
            #                      data['input_relpath'])
            test_dataset.assemble('_out', out_chan, output,
                                  data['target_relpath'])
            #test_dataset.assemble('_tgt', out_chan, target,
            #                      data['target_relpath'])
