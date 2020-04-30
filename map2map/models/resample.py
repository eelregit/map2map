import torch.nn as nn
import torch.nn.functional as F

from .narrow import narrow_by


def resample(x, scale_factor, narrow=True):
    modes = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}
    ndim = x.dim() - 2
    mode = modes[ndim]

    x = F.interpolate(x, scale_factor=scale_factor,
                      mode=mode, align_corners=False)

    if scale_factor > 1 and narrow == True:
        edges = round(scale_factor) // 2
        edges = max(edges, 1)
        x = narrow_by(x, edges)

    return x


class Resampler(nn.Module):
    """Resampling, upsampling or downsampling.

    By default discard the inaccurate edges when upsampling.
    """
    def __init__(self, ndim, scale_factor, narrow=True):
        super().__init__()

        modes = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}
        self.mode = modes[ndim]

        self.scale_factor = scale_factor
        self.narrow = narrow

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor,
                          mode=self.mode, align_corners=False)

        if self.scale_factor > 1 and self.narrow == True:
            edges = round(self.scale_factor) // 2
            edges = max(edges, 1)
            x = narrow_by(x, edges)

        return x
