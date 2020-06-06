from pprint import pprint
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader

from .data import FieldDataset
from . import models
from .models import narrow_like
from .state import load_model_state_dict


def test(args):
    pprint(vars(args))
    sys.stdout.flush()

    test_dataset = FieldDataset(
        in_patterns=args.test_in_patterns,
        tgt_patterns=args.test_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        augment=False,
        aug_add=None,
        aug_mul=None,
        crop=args.crop,
        pad=args.pad,
        scale_factor=args.scale_factor,
        cache=args.cache,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batches,
        shuffle=False,
        num_workers=args.loader_workers,
    )

    in_chan, out_chan = test_dataset.in_chan, test_dataset.tgt_chan

    model = getattr(models, args.model)
    model = model(sum(in_chan), sum(out_chan))
    criterion = getattr(torch.nn, args.criterion)
    criterion = criterion()

    device = torch.device('cpu')
    state = torch.load(args.load_state, map_location=device)
    load_model_state_dict(model, state['model'], strict=args.load_state_strict)
    print('model state at epoch {} loaded from {}'.format(
        state['epoch'], args.load_state))
    del state

    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            output = model(input)
            if args.pad > 0:  # FIXME
                output = narrow_like(output, target)
                input = narrow_like(input, target)
            else:
                target = narrow_like(target, output)
                input = narrow_like(input, output)

            loss = criterion(output, target)

            print('sample {} loss: {}'.format(i, loss.item()))

            if args.in_norms is not None:
                start = 0
                for norm, stop in zip(test_dataset.in_norms, np.cumsum(in_chan)):
                    norm(input[:, start:stop], undo=True)
                    start = stop
            if args.tgt_norms is not None:
                start = 0
                for norm, stop in zip(test_dataset.tgt_norms, np.cumsum(out_chan)):
                    norm(output[:, start:stop], undo=True)
                    norm(target[:, start:stop], undo=True)
                    start = stop

            np.savez('{}.npz'.format(i), input=input.numpy(),
                    output=output.numpy(), target=target.numpy())
