import os
import sys
import warnings
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FieldDataset
from .data import norms
from . import models
from .models import narrow_cast
from .models.gen_lin_field import gen_lin_field
from .utils import import_attr, load_model_state_dict


def generate(args):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            warnings.warn('Not parallelized but given more than 1 GPUs')

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device('cuda', 0)

        torch.backends.cudnn.benchmark = True
    else:  # CPU multithreading
        device = torch.device('cpu')

        if args.num_threads is None:
            args.num_threads = int(os.environ['SLURM_CPUS_ON_NODE'])

        torch.set_num_threads(args.num_threads)

    print('pytorch {}'.format(torch.__version__))
    pprint(vars(args))
    sys.stdout.flush()

    # Read power spectrum file and generate random linear field
    box_length = 1.e3 / 512 * args.num_mesh_1d
    ps_k, ps_p = np.loadtxt(args.power_spectrum, unpack=True)
    lin_field, seed = gen_lin_field(ps_k, ps_p, args.num_mesh_1d, box_length, z = args.redshift, seed = args.seed,
                                    sphere_mode = False)
    lin_filename = "./" + args.out_dir + "/lin.npy"
    np.save(lin_filename, np.float32(lin_field))
    del(lin_field)
    del(ps_k)
    del(ps_p)

    generate_dataset = FieldDataset(
        style_pattern=args.gen_style_pattern,
        in_patterns=[lin_filename],
        in_norms = None,
        tgt_patterns=[lin_filename],
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

    generate_loader = DataLoader(
        generate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    style_size = generate_dataset.style_size
    in_chan = generate_dataset.in_chan
    out_chan = generate_dataset.tgt_chan

    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(style_size, sum(in_chan), sum(out_chan),
                  scale_factor=args.scale_factor, **args.misc_kwargs)
    model.to(device)

    criterion = import_attr(args.criterion, torch.nn, models,
                            callback_at=args.callback_at)
    criterion = criterion()
    criterion.to(device)

    state = torch.load(args.load_state, map_location=device)
    load_model_state_dict(model, state['model'], strict=args.load_state_strict)
    print('model state at epoch {} loaded from {}'.format(
        state['epoch'], args.load_state))
    del state

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(generate_loader):

            style, input = data['style'], data['input']
            style = style.to(device, non_blocking=True)
            input = input.to(device, non_blocking=True)

            output = model(input, style)

            if i == 0 :
                print('##### sample :', i)
                print('style shape :', style.shape)
                print('input shape :', input.shape)
                print('output shape :', output.shape)

            input, output = narrow_cast(input, output)

            if i == 0 :
                print('narrowed shape :', output.shape, flush=True)

            norms.cosmology.dis(output, undo=True, **args.misc_kwargs)
            generate_dataset.assemble('dis_out', out_chan, output, [[args.out_dir + "/"]])
