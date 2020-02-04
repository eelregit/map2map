from math import log2, log10, ceil
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.cm import ScalarMappable


def fig3d(*fields, size=64, cmap=None, norm=None):
    fields = [f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f
            for f in fields]

    assert all(isinstance(f, np.ndarray) for f in fields)

    nc = fields[-1].shape[0]
    nf = len(fields)

    fig, axes = plt.subplots(nc, nf, squeeze=False, figsize=(5 * nf, 4.25 * nc))

    if cmap is None:
        if (fields[-1] >= 0).all():
            cmap = 'viridis'
        elif (fields[-1] <= 0).all():
            raise NotImplementedError
        else:
            cmap = 'RdBu_r'

    if norm is None:
        def quantize(x):
            return 2 ** round(log2(x), ndigits=1)

        l2, l1, h1, h2 = np.percentile(fields[-1], [2.5, 16, 84, 97.5])
        w1, w2 = (h1 - l1) / 2, (h2 - l2) / 2

        if (fields[-1] >= 0).all():
            if h1 > 0.1 * h2:
                norm = Normalize(vmin=0, vmax=quantize(2 * h2))
            else:
                norm = LogNorm(vmin=quantize(0.5 * l2), vmax=quantize(2 * h2))
        elif (fields[-1] <= 0).all():
            raise NotImplementedError
        else:
            if w1 > 0.1 * w2:
                vlim = quantize(2.5 * w1)
                norm = Normalize(vmin=-vlim, vmax=vlim)
            else:
                vlim = quantize(w2)
                norm = SymLogNorm(linthresh=0.1 * w1, vmin=-vlim, vmax=vlim)

    for c in range(nc):
        for f in range(nf):
            axes[c, f].imshow(fields[f][c, 0, :size, :size], cmap=cmap, norm=norm)
    plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes)

    return fig
