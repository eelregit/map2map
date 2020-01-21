import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FieldDataset
from . import models
from .models import narrow_like


def test(args):
    print(args)

    test_dataset = FieldDataset(
        in_patterns=args.test_in_patterns,
        tgt_patterns=args.test_tgt_patterns,
        augment=False,
        **vars(args),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batches,
        shuffle=False,
        num_workers=args.loader_workers,
    )

    in_channels, out_channels = test_dataset.channels

    model = getattr(models, args.model)
    model = model(in_channels, out_channels)
    criterion = getattr(torch.nn, args.criterion)
    criterion = criterion()

    device = torch.device('cpu')
    state = torch.load(args.load_state, map_location=device)
    model.load_state_dict(state['model'])
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

            if args.norms is not None:
                norm = test_dataset.norms[0]  # FIXME
                norm(input, undo=True)
                norm(output, undo=True)
                norm(target, undo=True)

            np.savez('{}.npz'.format(i), input=input.numpy(),
                    output=output.numpy(), target=target.numpy())
