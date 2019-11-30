from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from . import norms


class FieldDataset(Dataset):
    """Dataset of lists of fields.

    `in_patterns` is a list of glob patterns for the input fields.
    For example, `in_patterns=['/train/field1_*.npy', '/train/field2_*.npy']`.
    Likewise `tgt_patterns` is for target fields.
    Input and target samples of all fields are matched by sorting the globbed files.

    Data augmentations are supported for scalar and vector fields.

    `normalize` can be a list of callables to normalize each field.
    """
    def __init__(self, in_patterns, tgt_patterns, augment=False,
            normalize=None, **kwargs):
        in_file_lists = [sorted(glob(p)) for p in in_patterns]
        self.in_files = list(zip(* in_file_lists))

        tgt_file_lists = [sorted(glob(p)) for p in tgt_patterns]
        self.tgt_files = list(zip(* tgt_file_lists))

        assert len(self.in_files) == len(self.tgt_files), \
                'input and target sample sizes do not match'

        self.augment = augment

        self.normalize = normalize
        if self.normalize is not None:
            assert len(in_patterns) == len(self.normalize), \
                    'numbers of normalization callables and input fields do not match'

#        self.__dict__.update(kwargs)

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, idx):
        in_fields = [torch.from_numpy(np.load(f)).to(torch.float32)
                        for f in self.in_files[idx]]
        tgt_fields = [torch.from_numpy(np.load(f)).to(torch.float32)
                        for f in self.tgt_files[idx]]

        if self.augment:
            flip_axes = torch.randint(2, (3,), dtype=torch.bool)
            flip_axes = torch.arange(3)[flip_axes]

            flip3d(in_fields, flip_axes)
            flip3d(tgt_fields, flip_axes)

            perm_axes = torch.randperm(3)

            perm3d(in_fields, perm_axes)
            perm3d(tgt_fields, perm_axes)

        if self.normalize is not None:
            def get_norm(path):
                path = path.split('.')
                norm = norms
                while path:
                    norm = norm.__dict__[path.pop(0)]
                return norm

            for norm, ifield, tfield in zip(self.normalize, in_fields, tgt_fields):
                if isinstance(norm, str):
                    norm = get_norm(norm)

                norm(ifield)
                norm(tfield)

        in_fields = torch.cat(in_fields, dim=0)
        tgt_fields = torch.cat(tgt_fields, dim=0)

        return in_fields, tgt_fields


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
