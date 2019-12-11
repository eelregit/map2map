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

    Input fields can be padded (>0) or cropped (<0) with `pad_or_crop`.
    Padding assumes periodic boundary condition.

    Data augmentations are supported for scalar and vector fields.

    `norms` can be a list of callables to normalize each field.
    """
    def __init__(self, in_patterns, tgt_patterns, pad_or_crop=0, augment=False,
            norms=None):
        in_file_lists = [sorted(glob(p)) for p in in_patterns]
        self.in_files = list(zip(* in_file_lists))

        tgt_file_lists = [sorted(glob(p)) for p in tgt_patterns]
        self.tgt_files = list(zip(* tgt_file_lists))

        assert len(self.in_files) == len(self.tgt_files), \
                'input and target sample sizes do not match'

        self.in_channels = sum(np.load(f).shape[0] for f in self.in_files[0])
        self.tgt_channels = sum(np.load(f).shape[0] for f in self.tgt_files[0])

        if isinstance(pad_or_crop, int):
            pad_or_crop = (pad_or_crop,) * 6
        assert isinstance(pad_or_crop, tuple) and len(pad_or_crop) == 6, \
                'pad or crop size must be int or 6-tuple'
        self.pad_or_crop = np.array((0,) * 2 + pad_or_crop).reshape(4, 2)

        self.augment = augment

        if norms is not None:
            assert len(in_patterns) == len(norms), \
                    'numbers of normalization callables and input fields do not match'
            norms = [import_norm(norm) for norm in norms if isinstance(norm, str)]
        self.norms = norms

    @property
    def channels(self):
        return self.in_channels, self.tgt_channels

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, idx):
        in_fields = [np.load(f) for f in self.in_files[idx]]
        tgt_fields = [np.load(f) for f in self.tgt_files[idx]]

        padcrop(in_fields, self.pad_or_crop)  # with numpy

        in_fields = [torch.from_numpy(f).to(torch.float32) for f in in_fields]
        tgt_fields = [torch.from_numpy(f).to(torch.float32) for f in tgt_fields]

        if self.augment:
            flip_axes = torch.randint(2, (3,), dtype=torch.bool)
            flip_axes = torch.arange(3)[flip_axes]

            flip3d(in_fields, flip_axes)
            flip3d(tgt_fields, flip_axes)

            perm_axes = torch.randperm(3)

            perm3d(in_fields, perm_axes)
            perm3d(tgt_fields, perm_axes)

        if self.norms is not None:
            for norm, ifield, tfield in zip(self.norms, in_fields, tgt_fields):
                norm(ifield)
                norm(tfield)

        in_fields = torch.cat(in_fields, dim=0)
        tgt_fields = torch.cat(tgt_fields, dim=0)

        return in_fields, tgt_fields


def padcrop(fields, width):
    for i, x in enumerate(fields):
        if (width >= 0).all():
            x = np.pad(x, width, mode='wrap')
        elif (width <= 0).all():
            x = x[...,
                -width[0, 0] : width[0, 1],
                -width[1, 0] : width[1, 1],
                -width[2, 0] : width[2, 1],
            ]
        else:
            raise NotImplementedError('mixed pad-and-crop not supported')

        fields[i] = x


def flip3d(fields, axes):
    for i, x in enumerate(fields):
        if x.size(0) == 3:  # flip vector components
            x[axes] = - x[axes]

        axes = (1 + axes).tolist()
        x = torch.flip(x, axes)

        fields[i] = x


def perm3d(fields, axes):
    for i, x in enumerate(fields):
        if x.size(0) == 3:  # permutate vector components
            x = x[axes]

        axes = [0] + (1 + axes).tolist()
        x = x.permute(axes)

        fields[i] = x
