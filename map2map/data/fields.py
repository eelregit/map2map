import os
import pathlib
from glob import glob
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

    NOTE that vector fields are assumed if numbers of channels and dimensions are equal.

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
    """
    def __init__(self, in_patterns, tgt_patterns,
                 in_norms=None, tgt_norms=None, callback_at=None,
                 augment=False, aug_shift=None, aug_add=None, aug_mul=None,
                 crop=None, crop_start=None, crop_stop=None, crop_step=None,
                 in_pad=0, tgt_pad=0, scale_factor=1,
                 **kwargs):
        in_file_lists = [sorted(glob(p)) for p in in_patterns]
        self.in_files = list(zip(* in_file_lists))

        tgt_file_lists = [sorted(glob(p)) for p in tgt_patterns]
        self.tgt_files = list(zip(* tgt_file_lists))

        if len(self.in_files) != len(self.tgt_files):
            raise ValueError('number of input and target fields do not match')
        self.nfile = len(self.in_files)

        if self.nfile == 0:
            raise FileNotFoundError('file not found for {}'.format(in_patterns))
        self.is_read_once = np.full(self.nfile, False)

        self.in_chan = [np.load(f, mmap_mode='r').shape[0]
                        for f in self.in_files[0]]
        self.tgt_chan = [np.load(f, mmap_mode='r').shape[0]
                         for f in self.tgt_files[0]]

        self.size = np.load(self.in_files[0][0], mmap_mode='r').shape[1:]
        self.size = np.asarray(self.size)
        self.ndim = len(self.size)

        if in_norms is not None and len(in_patterns) != len(in_norms):
            raise ValueError('numbers of input normalization functions and fields do not match')
        self.in_norms = in_norms

        if tgt_norms is not None and len(tgt_patterns) != len(tgt_norms):
            raise ValueError('numbers of target normalization functions and fields do not match')
        self.tgt_norms = tgt_norms

        self.callback_at = callback_at

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
        self.crop_step = crop_step

        self.anchors = np.stack(np.mgrid[tuple(
            slice(crop_start[d], crop_stop[d], crop_step[d])
            for d in range(self.ndim)
        )], axis=-1).reshape(-1, self.ndim)
        self.ncrop = len(self.anchors)

        def format_pad(pad, ndim):
            if isinstance(pad, int):
                pad = np.broadcast_to(pad, ndim * 2)
            elif isinstance(pad, tuple) and len(pad) == ndim:
                pad = np.repeat(pad, 2)
            elif isinstance(pad, tuple) and len(pad) == ndim * 2:
                pad = np.array(pad)
            else:
                raise ValueError('pad and ndim mismatch')
            return pad.reshape(ndim, 2)
        self.in_pad = format_pad(in_pad, self.ndim)
        self.tgt_pad = format_pad(tgt_pad, self.ndim)

        if scale_factor != 1:
            tgt_size = np.load(self.tgt_files[0][0], mmap_mode='r').shape[1:]
            if any(self.size * scale_factor != tgt_size):
                raise ValueError('input size x scale factor != target size')
        self.scale_factor = scale_factor

        self.nsample = self.nfile * self.ncrop

        self.kwargs = kwargs

        self.assembly_line = {}

        self.commonpath = os.path.commonpath(
            file
            for files in self.in_files[:2] + self.tgt_files[:2]
            for file in files
        )

    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        ifile, icrop = divmod(idx, self.ncrop)

        # use memmap after reading a file once
        if self.is_read_once[ifile]:
            mmap_mode = 'r'
        else:
            mmap_mode = None
            self.is_read_once[ifile] = True

        in_fields = [np.load(f, mmap_mode=mmap_mode)
                     for f in self.in_files[ifile]]
        tgt_fields = [np.load(f, mmap_mode=mmap_mode)
                      for f in self.tgt_files[ifile]]

        anchor = self.anchors[icrop]

        for d, shift in enumerate(self.aug_shift):
            if shift is not None:
                anchor[d] += torch.randint(int(shift), (1,))

        # crop and pad are for the shapes after perm()
        # so before that they themselves need perm() in the opposite ways
        if self.augment:
            # let i and j index axes before and after perm()
            # then perm_axes is i_j, whose argsort is j_i
            # the latter is needed to index crop and pad for opposite perm()
            perm_axes = perm([], None, self.ndim)
            argsort_perm_axes = np.argsort(perm_axes.numpy())
        else:
            argsort_perm_axes = slice(None)

        crop(in_fields, anchor,
             self.crop[argsort_perm_axes],
             self.in_pad[argsort_perm_axes])
        crop(tgt_fields, anchor * self.scale_factor,
             self.crop[argsort_perm_axes] * self.scale_factor,
             self.tgt_pad[argsort_perm_axes])

        in_fields = [torch.from_numpy(f).to(torch.float32) for f in in_fields]
        tgt_fields = [torch.from_numpy(f).to(torch.float32) for f in tgt_fields]

        if self.in_norms is not None:
            for norm, x in zip(self.in_norms, in_fields):
                norm = import_attr(norm, norms, callback_at=self.callback_at)
                norm(x, **self.kwargs)
        if self.tgt_norms is not None:
            for norm, x in zip(self.tgt_norms, tgt_fields):
                norm = import_attr(norm, norms, callback_at=self.callback_at)
                norm(x, **self.kwargs)

        if self.augment:
            flip_axes = flip(in_fields, None, self.ndim)
            flip_axes = flip(tgt_fields, flip_axes, self.ndim)

            perm_axes = perm(in_fields, perm_axes, self.ndim)
            perm_axes = perm(tgt_fields, perm_axes, self.ndim)

        if self.aug_add is not None:
            add_fac = add(in_fields, None, self.aug_add)
            add_fac = add(tgt_fields, add_fac, self.aug_add)

        if self.aug_mul is not None:
            mul_fac = mul(in_fields, None, self.aug_mul)
            mul_fac = mul(tgt_fields, mul_fac, self.aug_mul)

        in_fields = torch.cat(in_fields, dim=0)
        tgt_fields = torch.cat(tgt_fields, dim=0)

        #in_relpath = [os.path.relpath(file, start=self.commonpath)
        #              for file in self.in_files[ifile]]
        tgt_relpath = [os.path.relpath(file, start=self.commonpath)
                       for file in self.tgt_files[ifile]]

        return {
            'input': in_fields,
            'target': tgt_fields,
            #'input_relpath': in_relpath,
            'target_relpath': tgt_relpath,
        }

    def assemble(self, label, chan, patches, paths):
        """Assemble and write whole fields from patches, each being the end
        result from a cropped field by `__getitem__`.

        Repeat feeding spatially ordered field patches.
        After filled, the whole fields are assembled and saved to relative
        paths specified by `paths` and `label`.
        `chan` is used to split the channels to undo `cat` in `__getitem__`.

        As an example, patches of shape `(1, 4, X, Y, Z)`, `label='_out'`
        and `chan=[1, 3]`, with `paths=[['d/scalar.npy'], ['d/vector.npy']]`
        will write to `'d/scalar_out.npy'` and `'d/vector_out.npy'`.

        Note that `paths` assumes transposed shape due to pytorch auto batching
        """
        if self.scale_factor != 1:
            raise NotImplementedError

        if isinstance(patches, torch.Tensor):
            patches = patches.detach().cpu().numpy()

        assert (patches.ndim == 2 + self.ndim,
                f'ndim mismatch: {patches.ndim, 2 + self.ndim}')
        if any(self.crop_step > patches.shape[2:]):
            raise RuntimeError('patch too small to tile')

        # the batched paths are a list of lists with shape (channel, batch)
        # since pytorch default_collate batches list of strings transposedly
        # therefore we transpose below back to (batch, channel)
        assert (patches.shape[1] == sum(chan),
                f'number of channels mismatch: {patches.shape[1], sum(chan)}')
        assert (len(paths) == len(chan),
                f'number of fields mismatch: {len(paths), len(chan)}')
        paths = list(zip(* paths))
        assert (patches.shape[0] == len(paths),
                f'batch size mismatch: {patches.shape[0], len(paths)}')

        patches = list(patches)
        if label in self.assembly_line:
            self.assembly_line[label] += patches
            self.assembly_line[label + 'path'] += paths
        else:
            self.assembly_line[label] = patches
            self.assembly_line[label + 'path'] = paths

        del patches, paths
        patches = self.assembly_line[label]
        paths = self.assembly_line[label + 'path']

        # NOTE anchor positioning assumes sufficient target padding and
        # symmetric narrowing (more on the right if odd) see `models/narrow.py`
        narrow = self.crop + self.tgt_pad.sum(axis=1) - patches[0].shape[1:]
        anchors = self.anchors - self.tgt_pad[:, 0] + narrow // 2

        while len(patches) >= self.ncrop:
            fields = np.zeros(patches[0].shape[:1] + tuple(self.size),
                              patches[0].dtype)

            for patch, anchor in zip(patches, anchors):
                fill(fields, patch, anchor)

            for field, path in zip(
                    np.split(fields, np.cumsum(chan), axis=0),
                    paths[0]):
                pathlib.Path(os.path.dirname(path)).mkdir(parents=True,
                                                          exist_ok=True)

                path = label.join(os.path.splitext(path))
                np.save(path, field)

            del patches[:self.ncrop], paths[:self.ncrop]


def fill(field, patch, anchor):
    ndim = len(anchor)
    assert (field.ndim == patch.ndim == 1 + ndim,
            f'ndim mismatch: {field.ndim, patch.ndim, 1 + ndim}')

    ind = [slice(None)]
    for d, (p, a, s) in enumerate(zip(
            patch.shape[1:], anchor, field.shape[1:])):
        i = np.arange(a, a + p)
        i %= s
        i = i.reshape((-1,) + (1,) * (ndim - d - 1))
        ind.append(i)
    ind = tuple(ind)

    field[ind] = patch


def crop(fields, anchor, crop, pad):
    assert (all(x.shape[1:] == fields[0].shape[1:] for x in fields),
            f'shape mismatch: {[x.shape[1:] for x in fields]}')
    size = fields[0].shape[1:]
    ndim = len(size)
    assert (ndim == len(anchor) == len(crop) == len(pad),
            f'ndim mismatch: {ndim, len(anchor), len(crop), len(pad)}')

    ind = [slice(None)]
    for d, (a, c, (p0, p1), s) in enumerate(zip(anchor, crop, pad, size)):
        i = np.arange(a - p0, a + c + p1)
        i %= s
        i = i.reshape((-1,) + (1,) * (ndim - d - 1))
        ind.append(i)
    ind = tuple(ind)

    for i, x in enumerate(fields):
        x = x[ind]

        fields[i] = x

    return ind


def flip(fields, axes, ndim):
    assert ndim > 1, 'flipping is ambiguous for 1D scalars/vectors'

    if axes is None:
        axes = torch.randint(2, (ndim,), dtype=torch.bool)
        axes = torch.arange(ndim)[axes]

    for i, x in enumerate(fields):
        if x.shape[0] == ndim:  # flip vector components
            x[axes] = - x[axes]

        shifted_axes = (1 + axes).tolist()
        x = torch.flip(x, shifted_axes)

        fields[i] = x

    return axes


def perm(fields, axes, ndim):
    assert ndim > 1, 'permutation is not necessary for 1D fields'

    if axes is None:
        axes = torch.randperm(ndim)

    for i, x in enumerate(fields):
        if x.shape[0] == ndim:  # permutate vector components
            x = x[axes]

        shifted_axes = [0] + (1 + axes).tolist()
        x = x.permute(shifted_axes)

        fields[i] = x

    return axes


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
