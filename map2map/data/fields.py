from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .norms import import_norm


class FieldDataset(Dataset):
    """Dataset of lists of fields.

    `in_patterns` is a list of glob patterns for the input fields.
    For example, `in_patterns=['/train/field1_*.npy', '/train/field2_*.npy']`.
    Likewise `tgt_patterns` is for target fields.
    Input and target samples are matched by sorting the globbed files.

    `in_norms` is a list of of functions to normalize the input fields.
    Likewise for `tgt_norms`.

    Data augmentations are supported for scalar and vector fields.

    Input and target fields can be cropped.
    Input fields can be padded assuming periodic boundary condition.

    Setting integer `scale_factor` greater than 1 will crop target bigger than
    the input for super-resolution, in which case `crop` and `pad` are sizes of
    the input resolution.

    `cache` enables data caching.
    `div_data` enables data division, useful when combined with caching.
    """
    def __init__(self, in_patterns, tgt_patterns,
                 in_norms=None, tgt_norms=None,
                 augment=False, crop=None, pad=0, scale_factor=1,
                 cache=False, div_data=False, rank=None, world_size=None):
        in_file_lists = [sorted(glob(p)) for p in in_patterns]
        self.in_files = list(zip(* in_file_lists))

        tgt_file_lists = [sorted(glob(p)) for p in tgt_patterns]
        self.tgt_files = list(zip(* tgt_file_lists))

        assert len(self.in_files) == len(self.tgt_files), \
                'input and target sample sizes do not match'

        assert len(self.in_files) > 0, 'file not found for {}'.format(in_patterns)

        if div_data:
            files = len(self.in_files) // world_size
            self.in_files = self.in_files[rank * files : (rank + 1) * files]
            self.tgt_files = self.tgt_files[rank * files : (rank + 1) * files]

        self.in_chan = [np.load(f).shape[0] for f in self.in_files[0]]
        self.tgt_chan = [np.load(f).shape[0] for f in self.tgt_files[0]]

        self.size = np.load(self.in_files[0][0]).shape[1:]
        self.size = np.asarray(self.size)
        self.ndim = len(self.size)

        if in_norms is not None:
            assert len(in_patterns) == len(in_norms), \
                    'numbers of input normalization functions and fields do not match'
            in_norms = [import_norm(norm) for norm in in_norms]
        self.in_norms = in_norms

        if tgt_norms is not None:
            assert len(tgt_patterns) == len(tgt_norms), \
                    'numbers of target normalization functions and fields do not match'
            tgt_norms = [import_norm(norm) for norm in tgt_norms]
        self.tgt_norms = tgt_norms

        self.augment = augment
        if self.ndim == 1 and self.augment:
            raise ValueError('cannot augment 1D fields')

        if crop is None:
            self.crop = self.size
            self.reps = np.ones_like(self.size)
        else:
            self.crop = np.broadcast_to(crop, self.size.shape)
            self.reps = self.size // self.crop
        self.tot_reps = int(np.prod(self.reps))

        assert isinstance(pad, int), 'only support symmetric padding for now'
        self.pad = np.broadcast_to(pad, (self.ndim, 2))

        assert isinstance(scale_factor, int) and scale_factor >= 1, \
                "only support integer upsampling"
        self.scale_factor = scale_factor

        self.cache = cache
        if self.cache:
            self.in_fields = {}
            self.tgt_fields = {}

    def __len__(self):
        return len(self.in_files) * self.tot_reps

    def __getitem__(self, idx):
        idx, sub_idx = idx // self.tot_reps, idx % self.tot_reps
        start = np.unravel_index(sub_idx, self.reps) * self.crop

        if self.cache:
            try:
                in_fields = self.in_fields[idx]
                tgt_fields = self.tgt_fields[idx]
            except KeyError:
                in_fields = [np.load(f) for f in self.in_files[idx]]
                tgt_fields = [np.load(f) for f in self.tgt_files[idx]]
                self.in_fields[idx] = in_fields
                self.tgt_fields[idx] = tgt_fields
        else:
            in_fields = [np.load(f) for f in self.in_files[idx]]
            tgt_fields = [np.load(f) for f in self.tgt_files[idx]]

        in_fields = crop(in_fields, start, self.crop, self.pad)
        tgt_fields = crop(tgt_fields, start * self.scale_factor,
                          self.crop * self.scale_factor,
                          np.zeros_like(self.pad))

        in_fields = [torch.from_numpy(f).to(torch.float32) for f in in_fields]
        tgt_fields = [torch.from_numpy(f).to(torch.float32) for f in tgt_fields]

        if self.augment:
            flip_axes = torch.randint(2, (self.ndim,), dtype=torch.bool)
            flip_axes = torch.arange(self.ndim)[flip_axes]

            in_fields = flip(in_fields, flip_axes, self.ndim)
            tgt_fields = flip(tgt_fields, flip_axes, self.ndim)

            perm_axes = torch.randperm(self.ndim)

            in_fields = perm(in_fields, perm_axes, self.ndim)
            tgt_fields = perm(tgt_fields, perm_axes, self.ndim)

        if self.in_norms is not None:
            for norm, x in zip(self.in_norms, in_fields):
                norm(x)
        if self.tgt_norms is not None:
            for norm, x in zip(self.tgt_norms, tgt_fields):
                norm(x)

        in_fields = torch.cat(in_fields, dim=0)
        tgt_fields = torch.cat(tgt_fields, dim=0)

        return in_fields, tgt_fields


def crop(fields, start, crop, pad):
    new_fields = []
    for x in fields:
        for d, (i, c, (p0, p1)) in enumerate(zip(start, crop, pad)):
            begin, end = i - p0, i + c + p1
            x = x.take(range(begin, end), axis=1 + d, mode='wrap')

        new_fields.append(x)

    return new_fields


def flip(fields, axes, ndim):
    assert ndim > 1, 'flipping is ambiguous for 1D vectors'

    new_fields = []
    for x in fields:
        if x.shape[0] == ndim:  # flip vector components
            x[axes] = - x[axes]

        shifted_axes = (1 + axes).tolist()
        x = torch.flip(x, shifted_axes)

        new_fields.append(x)

    return new_fields


def perm(fields, axes, ndim):
    assert ndim > 1, 'permutation is not necessary for 1D fields'

    new_fields = []
    for x in fields:
        if x.shape[0] == ndim:  # permutate vector components
            x = x[axes]

        shifted_axes = [0] + (1 + axes).tolist()
        x = x.permute(shifted_axes)

        new_fields.append(x)

    return new_fields
