from math import log2, log10, ceil
import warnings
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.cm import ScalarMappable


def quantize(x):
    return 2 ** round(log2(x), ndigits=1)


def plt_slices(*fields, size=64, title=None, cmap=None, norm=None):
    """Plot slices of fields of more than 2 spatial dimensions.
    """
    plt.close('all')

    fields = [field.detach().cpu().numpy() if isinstance(field, torch.Tensor)
            else field for field in fields]

    assert all(isinstance(field, np.ndarray) for field in fields)
    assert all(field.ndim == fields[0].ndim for field in fields)

    nc = max(field.shape[0] for field in fields)
    nf = len(fields)

    if title is not None:
        assert len(title) == nf
    cmap = np.broadcast_to(cmap, (nf,))
    norm = np.broadcast_to(norm, (nf,))

    im_size = 2
    cbar_height = 0.2
    fig, axes = plt.subplots(
        nc + 1, nf,
        squeeze=False,
        figsize=(nf * im_size, nc * im_size + cbar_height),
        dpi=100,
        gridspec_kw={'height_ratios': nc * [im_size] + [cbar_height]}
    )

    for f, (field, cmap_col, norm_col) in enumerate(zip(fields, cmap, norm)):
        all_non_neg = (field >= 0).all()
        all_non_pos = (field <= 0).all()

        if cmap_col is None:
            if all_non_neg:
                cmap_col = 'viridis'
            elif all_non_pos:
                warnings.warn('no implementation for all non-positive values')
                cmap_col = None
            else:
                cmap_col = 'RdBu_r'

        if norm_col is None:
            l2, l1, h1, h2 = np.percentile(field, [2.5, 16, 84, 97.5])
            w1, w2 = (h1 - l1) / 2, (h2 - l2) / 2

            if all_non_neg:
                if h1 > 0.1 * h2:
                    norm_col = Normalize(vmin=0, vmax=quantize(h2))
                else:
                    norm_col = LogNorm(vmin=quantize(l2), vmax=quantize(h2))
            elif all_non_pos:
                warnings.warn('no implementation for all non-positive values yet')
                norm_col = None
            else:
                vlim = quantize(max(-l2, h2))
                if w1 > 0.1 * w2 or l1 * h1 >= 0:
                    norm_col = Normalize(vmin=-vlim, vmax=vlim)
                else:
                    linthresh = 0.1 * quantize(min(-l1, h1))
                    norm_col = SymLogNorm(linthresh=linthresh, vmin=-vlim, vmax=vlim)

        for c in range(field.shape[0]):
            s0 = (c,) + tuple(d // 2 for d in field.shape[1:-2])
            s1 = (
                slice(
                    (field.shape[-2] - size) // 2,
                    (field.shape[-2] + size) // 2,
                ),
                slice(
                    (field.shape[-1] - size) // 2,
                    (field.shape[-1] + size) // 2,
                ),
            )

            axes[c, f].pcolormesh(field[s0 + s1], cmap=cmap_col, norm=norm_col)

            axes[c, f].set_aspect('equal')

            axes[c, f].set_xticks([])
            axes[c, f].set_yticks([])

            if c == 0 and title is not None:
                axes[c, f].set_title(title[f])

        for c in range(field.shape[0], nc):
            axes[c, f].axis('off')

        fig.colorbar(
            ScalarMappable(norm=norm_col, cmap=cmap_col),
            cax=axes[-1, f],
            orientation='horizontal',
        )

    fig.tight_layout()

    return fig
