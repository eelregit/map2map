from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from .norms import import_norm


class FieldDataset(Dataset):
    """Dataset of lists of fields.

    `in_patterns` is a list of glob patterns for the input fields.
    For example, `in_patterns=['/train/field1_*.npy', '/train/field2_*.npy']`.
    Likewise `tgt_patterns` is for target fields.
    Input and target samples of all fields are matched by sorting the globbed files.

    `norms` can be a list of callables to normalize each field.

    Data augmentations are supported for scalar and vector fields.

    Input and target fields can be cropped.
    Input fields can be padded assuming periodic boundary condition.

    `cache` enables data caching.
    `div_data` enables data division, useful when combined with caching.
    """
    def __init__(self, in_patterns, tgt_patterns,
            norms=None, augment=False, crop=None, pad=0,
            cache=False, div_data=False, rank=None, world_size=None,
            **kwargs):
        in_file_lists = [sorted(glob(p)) for p in in_patterns]
        self.in_files = list(zip(* in_file_lists))

        tgt_file_lists = [sorted(glob(p)) for p in tgt_patterns]
        self.tgt_files = list(zip(* tgt_file_lists))

        assert len(self.in_files) == len(self.tgt_files), \
                'input and target sample sizes do not match'

        if div_data:
            files = len(self.in_files) // world_size
            self.in_files = self.in_files[rank * files : (rank + 1) * files]
            self.tgt_files = self.tgt_files[rank * files : (rank + 1) * files]

        self.in_channels = sum(np.load(f).shape[0] for f in self.in_files[0])
        self.tgt_channels = sum(np.load(f).shape[0] for f in self.tgt_files[0])

        self.size = np.load(self.in_files[0][0]).shape[1:]
        self.size = np.asarray(self.size)
        self.ndim = len(self.size)

        if norms is not None:  # FIXME: in_norms, tgt_norms
            assert len(in_patterns) == len(norms), \
                    'numbers of normalization callables and input fields do not match'
            norms = [import_norm(norm) for norm in norms if isinstance(norm, str)]
        self.norms = norms

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

        self.cache = cache
        if self.cache:
            self.in_fields = {}
            self.tgt_fields = {}

    def __len__(self):
        return len(self.in_files) * self.tot_reps

    @property
    def channels(self):
        return self.in_channels, self.tgt_channels

    def __getitem__(self, idx):
        idx, sub_idx = idx // self.tot_reps, idx % self.tot_reps
        start = np.unravel_index(sub_idx, self.reps) * self.crop
        #print('==================================================')
        #print(f'idx = {idx}, sub_idx = {sub_idx}, start = {start}')
        #print(f'self.reps = {self.reps}, self.tot_reps = {self.tot_reps}')
        #print(f'self.crop = {self.crop}, self.size = {self.size}')
        #print(f'self.ndim = {self.ndim}, self.channels = {self.channels}')
        #print(f'self.pad = {self.pad}')

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
        tgt_fields = crop(tgt_fields, start, self.crop, np.zeros_like(self.pad))

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

        if self.norms is not None:
            for norm, ifield, tfield in zip(self.norms, in_fields, tgt_fields):
                norm(ifield)
                norm(tfield)

        in_fields = torch.cat(in_fields, dim=0)
        tgt_fields = torch.cat(tgt_fields, dim=0)
        #print(in_fields.shape, tgt_fields.shape)

        return in_fields, tgt_fields


def crop(fields, start, crop, pad):
    new_fields = []
    for x in fields:
        for d, (i, N, (p0, p1)) in enumerate(zip(start, crop, pad)):
            x = x.take(range(i - p0, i + N + p1), axis=1 + d, mode='wrap')

        new_fields.append(x)

    return new_fields


def flip(fields, axes, ndim):
    assert ndim > 1, 'flipping is ambiguous for 1D vectors'

    new_fields = []
    for x in fields:
        if x.size(0) == ndim:  # flip vector components
            x[axes] = - x[axes]

        axes = (1 + axes).tolist()
        x = torch.flip(x, axes)

        new_fields.append(x)

    return new_fields


def perm(fields, axes, ndim):
    assert ndim > 1, 'permutation is not necessary for 1D fields'

    new_fields = []
    for x in fields:
        if x.size(0) == ndim:  # permutate vector components
            x = x[axes]

        axes = [0] + (1 + axes).tolist()
        x = x.permute(axes)

        new_fields.append(x)

    return new_fields
