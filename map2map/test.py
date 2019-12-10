import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FieldDataset
from . import models
from .models import narrow_like


def test(args):
    test_dataset = FieldDataset(
        in_patterns=args.test_in_patterns,
        tgt_patterns=args.test_tgt_patterns,
        augment=False,
        norms=args.norms,
        pad_or_crop=args.pad_or_crop,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batches,
        shuffle=False,
        num_workers=args.loader_workers,
    )

    in_channels, out_channels = test_dataset.channels

    model = models.__dict__[args.model](in_channels, out_channels)
    criterion = torch.nn.__dict__[args.criterion]()

    device = torch.device('cpu')
    state = torch.load(args.load_state, map_location=device)
    from collections import OrderedDict
    model_state = OrderedDict()
    for k, v in state['model'].items():
        model_k = k.replace('module.', '', 1)  # FIXME
        model_state[model_k] = v
    model.load_state_dict(model_state)
    print('model state at epoch {} loaded from {}'.format(
        state['epoch'], args.load_state))
    del state

    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            output = model(input)
            if args.pad_or_crop > 0:  # FIXME
                output = narrow_like(output, target)
            else:
                target = narrow_like(target, output)

            loss = criterion(output, target)

            print('sample {} loss: {}'.format(i, loss.item()))

            if args.norms is not None:
                norm = test_dataset.norms[0]  # FIXME
                norm(output, undo=True)

            np.savez('{}.npz'.format(i), input=input.numpy(),
                    output=output.numpy(), target=target.numpy())
