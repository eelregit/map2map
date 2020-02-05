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

    colorbar_frac = 0.15 / (0.85 * nc + 0.15)
    fig, axes = plt.subplots(nc, nf, squeeze=False, figsize=(4 * nf, 4 * nc * (1 + colorbar_frac)))

    def quantize(x):
        return 2 ** round(log2(x), ndigits=1)

    for f in range(nf):
        all_non_neg = (fields[f] >= 0).all()
        all_non_pos = (fields[f] <= 0).all()

        if cmap is None:
            if all_non_neg:
                cmap_ = 'viridis'
            elif all_non_pos:
                raise NotImplementedError
            else:
                cmap_ = 'RdBu_r'
        else:
            cmap_ = cmap

        if norm is None:
            l2, l1, h1, h2 = np.percentile(fields[f], [2.5, 16, 84, 97.5])
            w1, w2 = (h1 - l1) / 2, (h2 - l2) / 2

            if all_non_neg:
                if h1 > 0.1 * h2:
                    norm_ = Normalize(vmin=0, vmax=quantize(2 * h2))
                else:
                    norm_ = LogNorm(vmin=quantize(0.5 * l2), vmax=quantize(2 * h2))
            elif all_non_pos:
                raise NotImplementedError
            else:
                if w1 > 0.1 * w2:
                    vlim = quantize(2.5 * w1)
                    norm_ = Normalize(vmin=-vlim, vmax=vlim)
                else:
                    vlim = quantize(w2)
                    norm_ = SymLogNorm(linthresh=0.1 * w1, vmin=-vlim, vmax=vlim)
        else:
            norm_ = norm

        for c in range(nc):
            axes[c, f].imshow(fields[f][c, 0, :size, :size], cmap=cmap_, norm=norm_)

        plt.colorbar(ScalarMappable(norm=norm_, cmap=cmap_), ax=axes[:, f],
                     orientation='horizontal', fraction=colorbar_frac, pad=0.05)

    return fig
