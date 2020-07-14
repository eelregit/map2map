from glob import glob
from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..utils import import_attr
from . import norms


class FieldDataset(Dataset):
    """Dataset of lists of fields.

    `in_patterns` is a list of glob patterns for the input field files.
    For example, `in_patterns=['/train/field1_*.npy', '/train/field2_*.npy']`.
    Each pattern in the list is a new field.
    Likewise `tgt_patterns` is for target fields.
    Input and target fields are matched by sorting the globbed files.

    `in_norms` is a list of of functions to normalize the input fields.
    Likewise for `tgt_norms`.

    Scalar and vector fields can be augmented by flipping and permutating the axes.
    In 3D these form the full octahedral symmetry, the Oh group of order 48.
    In 2D this is the dihedral group D4 of order 8.
    1D is not supported, but can be done easily by preprocessing.
    Fields can be augmented by random shift by a few pixels, useful for models
    that treat neighboring pixels differently, e.g. with strided convolutions.
    Additive and multiplicative augmentation are also possible, but with all fields
    added or multiplied by the same factor.

    Input and target fields can be cropped, to return multiple slices of size
    `crop` from each field.
    The crop anchors are controlled by `crop_start`, `crop_stop`, and `crop_step`.
    Input (but not target) fields can be padded beyond the crop size assuming
    periodic boundary condition.

    Setting integer `scale_factor` greater than 1 will crop target bigger than
    the input for super-resolution, in which case `crop` and `pad` are sizes of
    the input resolution.

    `cache` enables LRU cache of the input and target fields, up to `cache_maxsize`
    pairs (unlimited by default).
    `div_data` enables data division, to be used with `cache`, so that different
    fields are cached in different GPU processes.
    This saves CPU RAM but limits stochasticity.
    """
    def __init__(self, in_patterns, tgt_patterns,
                 in_norms=None, tgt_norms=None, callback_at=None,
                 augment=False, aug_shift=None, aug_add=None, aug_mul=None,
                 crop=None, crop_start=None, crop_stop=None, crop_step=None,
                 pad=0, scale_factor=1,
                 cache=False, cache_maxsize=None, div_data=False,
                 rank=None, world_size=None):
        in_file_lists = [sorted(glob(p)) for p in in_patterns]
        self.in_files = list(zip(* in_file_lists))

        tgt_file_lists = [sorted(glob(p)) for p in tgt_patterns]
        self.tgt_files = list(zip(* tgt_file_lists))

        assert len(self.in_files) == len(self.tgt_files), \
                'number of input and target fields do not match'
        self.nfile = len(self.in_files)

        assert self.nfile > 0, 'file not found for {}'.format(in_patterns)

        self.in_chan = [np.load(f).shape[0] for f in self.in_files[0]]
        self.tgt_chan = [np.load(f).shape[0] for f in self.tgt_files[0]]

        self.size = np.load(self.in_files[0][0]).shape[1:]
        self.size = np.asarray(self.size)
        self.ndim = len(self.size)

        if in_norms is not None:
            assert len(in_patterns) == len(in_norms), \
                    'numbers of input normalization functions and fields do not match'
            in_norms = [import_attr(norm, norms.__name__, callback_at)
                        for norm in in_norms]
        self.in_norms = in_norms

        if tgt_norms is not None:
            assert len(tgt_patterns) == len(tgt_norms), \
                    'numbers of target normalization functions and fields do not match'
            tgt_norms = [import_attr(norm, norms.__name__, callback_at)
                         for norm in tgt_norms]
        self.tgt_norms = tgt_norms

        self.augment = augment
        if self.ndim == 1 and self.augment:
            raise ValueError('cannot augment 1D fields')
        self.aug_shift = np.broadcast_to(aug_shift, (self.ndim,))
        self.aug_add = aug_add
        self.aug_mul = aug_mul

        if crop is None:
            self.crop = self.size
        else:
            self.crop = np.broadcast_to(crop, (self.ndim,))

        if crop_start is None:
            crop_start = np.zeros_like(self.size)
        else:
            crop_start = np.broadcast_to(crop_start, (self.ndim,))

        if crop_stop is None:
            crop_stop = self.size
        else:
            crop_stop = np.broadcast_to(crop_stop, (self.ndim,))

        if crop_step is None:
            crop_step = self.crop
        else:
            crop_step = np.broadcast_to(crop_step, (self.ndim,))

        self.anchors = np.stack(np.mgrid[tuple(
            slice(crop_start[d], crop_stop[d], crop_step[d])
            for d in range(self.ndim)
        )], axis=-1).reshape(-1, self.ndim)
        self.ncrop = len(self.anchors)

        assert isinstance(pad, int), 'only support symmetric padding for now'
        self.pad = np.broadcast_to(pad, (self.ndim, 2))

        assert isinstance(scale_factor, int) and scale_factor >= 1, \
                'only support integer upsampling'
        self.scale_factor = scale_factor

        if cache:
            self.get_fields = lru_cache(maxsize=cache_maxsize)(self.get_fields)

        if div_data:
            self.samples = []

            # first add full fields when num_fields > num_GPU
            for i in range(rank, self.nfile // world_size * world_size,
                           world_size):
                self.samples.extend(list(
                    range(i * self.ncrop, (i + 1) * self.ncrop)
                ))

            # then split the rest into fractions of fields
            # drop the last incomplete batch of samples
            frac_start = self.nfile // world_size * world_size * self.ncrop
            frac_samples = self.nfile % world_size * self.ncrop // world_size
            self.samples.extend(list(
                range(frac_start + rank * frac_samples,
                      frac_start + (rank + 1) * frac_samples)
            ))
        else:
            self.samples = list(range(self.nfile * self.ncrop))
        self.nsample = len(self.samples)

        self.rank = rank

    def get_fields(self, idx):
        in_fields = [np.load(f) for f in self.in_files[idx]]
        tgt_fields = [np.load(f) for f in self.tgt_files[idx]]
        return in_fields, tgt_fields

    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        idx = self.samples[idx]

        in_fields, tgt_fields = self.get_fields(idx // self.ncrop)

        anchor = self.anchors[idx % self.ncrop]

        for d, shift in enumerate(self.aug_shift):
            if shift is not None:
                anchor[d] += torch.randint(shift, (1,))

        in_fields = crop(in_fields, anchor, self.crop, self.pad)
        tgt_fields = crop(tgt_fields, anchor * self.scale_factor,
                          self.crop * self.scale_factor,
                          np.zeros_like(self.pad))

        in_fields = [torch.from_numpy(f).to(torch.float32) for f in in_fields]
        tgt_fields = [torch.from_numpy(f).to(torch.float32) for f in tgt_fields]

        if self.in_norms is not None:
            for norm, x in zip(self.in_norms, in_fields):
                norm(x)
        if self.tgt_norms is not None:
            for norm, x in zip(self.tgt_norms, tgt_fields):
                norm(x)

        if self.augment:
            in_fields, flip_axes = flip(in_fields, None, self.ndim)
            tgt_fields, flip_axes = flip(tgt_fields, flip_axes, self.ndim)

            in_fields, perm_axes = perm(in_fields, None, self.ndim)
            tgt_fields, perm_axes = perm(tgt_fields, perm_axes, self.ndim)

        if self.aug_add is not None:
            add_fac = add(in_fields, None, self.aug_add)
            add_fac = add(tgt_fields, add_fac, self.aug_add)

        if self.aug_mul is not None:
            mul_fac = mul(in_fields, None, self.aug_mul)
            mul_fac = mul(tgt_fields, mul_fac, self.aug_mul)

        in_fields = torch.cat(in_fields, dim=0)
        tgt_fields = torch.cat(tgt_fields, dim=0)

        return in_fields, tgt_fields


def crop(fields, anchor, crop, pad):
    new_fields = []
    for x in fields:
        for d, (a, c, (p0, p1)) in enumerate(zip(anchor, crop, pad)):
            begin, end = a - p0, a + c + p1
            x = x.take(range(begin, end), axis=1 + d, mode='wrap')

        new_fields.append(x)

    return new_fields


def flip(fields, axes, ndim):
    assert ndim > 1, 'flipping is ambiguous for 1D scalars/vectors'

    if axes is None:
        axes = torch.randint(2, (ndim,), dtype=torch.bool)
        axes = torch.arange(ndim)[axes]

    new_fields = []
    for x in fields:
        if x.shape[0] == ndim:  # flip vector components
            x[axes] = - x[axes]

        shifted_axes = (1 + axes).tolist()
        x = torch.flip(x, shifted_axes)

        new_fields.append(x)

    return new_fields, axes


def perm(fields, axes, ndim):
    assert ndim > 1, 'permutation is not necessary for 1D fields'

    if axes is None:
        axes = torch.randperm(ndim)

    new_fields = []
    for x in fields:
        if x.shape[0] == ndim:  # permutate vector components
            x = x[axes]

        shifted_axes = [0] + (1 + axes).tolist()
        x = x.permute(shifted_axes)

        new_fields.append(x)

    return new_fields, axes


def add(fields, fac, std):
    if fac is None:
        x = fields[0]
        fac = torch.zeros((x.shape[0],) + (1,) * (x.dim() - 1))
        fac.normal_(mean=0, std=std)

    for x in fields:
        x += fac

    return fac


def mul(fields, fac, std):
    if fac is None:
        x = fields[0]
        fac = torch.ones((x.shape[0],) + (1,) * (x.dim() - 1))
        fac.log_normal_(mean=0, std=std)

    for x in fields:
        x *= fac

    return fac
