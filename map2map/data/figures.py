from math import log2, log10, ceil
import warnings
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.cm import ScalarMappable


def plt_slices(*fields, size=64, title=None, cmap=None, norm=None):
    """Plot slices of fields of more than 2 spatial dimensions.
    """
    fields = [field.detach().cpu().numpy() if isinstance(field, torch.Tensor)
            else field for field in fields]

    assert all(isinstance(field, np.ndarray) for field in fields)
    assert all(field.ndim == fields[0].ndim for field in fields)

    nc = max(field.shape[0] for field in fields)
    nf = len(fields)
    nd = fields[0].ndim - 1

    if title is not None:
        assert len(title) == nf

    im_size = 2
    cbar_height = 0.3
    cbar_frac = cbar_height / (nc * im_size + cbar_height)
    fig, axes = plt.subplots(
        nc, nf,
        squeeze=False,
        figsize=(nf * im_size, nc * im_size + cbar_height),
        dpi=100,
        constrained_layout=True,
    )

    def quantize(x):
        return 2 ** round(log2(x), ndigits=1)

    for f, field in enumerate(fields):
        all_non_neg = (field >= 0).all()
        all_non_pos = (field <= 0).all()

        if cmap is None:
            if all_non_neg:
                cmap_ = 'viridis'
            elif all_non_pos:
                warnings.warn('no implementation for all non-positive values')
                cmap_ = None
            else:
                cmap_ = 'RdBu_r'
        else:
            cmap_ = cmap

        if norm is None:
            l2, l1, h1, h2 = np.percentile(field, [2.5, 16, 84, 97.5])
            w1, w2 = (h1 - l1) / 2, (h2 - l2) / 2

            if all_non_neg:
                if h1 > 0.1 * h2:
                    norm_ = Normalize(vmin=0, vmax=quantize(2 * h2))
                else:
                    norm_ = LogNorm(vmin=quantize(0.5 * l2), vmax=quantize(2 * h2))
            elif all_non_pos:
                warnings.warn('no implementation for all non-positive values')
                norm_ = None
            else:
                if w1 > 0.1 * w2:
                    vlim = quantize(2.5 * w1)
                    norm_ = Normalize(vmin=-vlim, vmax=vlim)
                else:
                    vlim = quantize(w2)
                    norm_ = SymLogNorm(linthresh=0.1 * w1, vmin=-vlim, vmax=vlim)
        else:
            norm_ = norm

        for c in range(field.shape[0]):
            s = (c,) + (0,) * (nd - 2) + (slice(64),) * 2
            axes[c, f].pcolormesh(field[s], cmap=cmap_, norm=norm_)

            axes[c, f].set_aspect('equal')

            axes[c, f].set_xticks([])
            axes[c, f].set_yticks([])

            if c == 0 and title is not None:
                axes[c, f].set_title(title[f])

        for c in range(field.shape[0], nc):
            axes[c, f].axis('off')

        fig.colorbar(
            ScalarMappable(norm=norm_, cmap=cmap_),
            ax=axes[:, f],
            orientation='horizontal',
            fraction=cbar_frac,
            pad=0,
            shrink=0.9,
            aspect=10,
        )

    fig.set_constrained_layout_pads(w_pad=2/72, h_pad=2/72, wspace=0, hspace=0)

    return fig
