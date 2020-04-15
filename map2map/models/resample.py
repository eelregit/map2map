from functools import partial
import torch.nn as nn
import torch.nn.functional as F


def get_resampler(ndim, scale_factor):
    modes = {1: 'linear', 2: 'bilinear', 3:'trilinear'}
    mode = modes[ndim]

    resampler = partial(F.interpolate, scale_factor=scale_factor,
            mode=mode, align_corners=False)

    return resampler


class Resampler(nn.Module):
    def __init__(self, ndim, scale_factor):
        super().__init__()
        self.resample = get_resampler(ndim, scale_factor)

    def forward(self, input):
        return self.resample(input)
