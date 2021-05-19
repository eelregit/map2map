from math import log2, log10, ceil
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.cm import ScalarMappable
plt.rc('text', usetex=False)

from ..models import lag2eul, power


def quantize(x):
    return 2 ** round(log2(x), ndigits=1)


def plt_slices(*fields, size=64, title=None, cmap=None, norm=None, **kwargs):
    """Plot slices of fields of more than 2 spatial dimensions.

    Each field should have a channel dimension followed by spatial dimensions,
    i.e. no batch dimension.
    """
    plt.close('all')

    assert all(isinstance(field, torch.Tensor) for field in fields)

    fields = [field.detach().cpu().numpy() for field in fields]

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
        all_non_neg = np.all(field >= 0)
        all_non_pos = np.all(field <= 0)

        if cmap_col is None:
            if all_non_neg:
                cmap_col = 'inferno'
            elif all_non_pos:
                cmap_col = 'inferno_r'
            else:
                cmap_col = 'RdBu_r'

        if norm_col is None:
            l2, l1, h1, h2 = np.percentile(field, [2.5, 16, 84, 97.5])
            w1, w2 = (h1 - l1) / 2, (h2 - l2) / 2

            if all_non_neg:
                if h1 > 0.1 * h2 or l2 == 0:
                    norm_col = Normalize(vmin=0, vmax=quantize(h2))
                else:
                    norm_col = LogNorm(vmin=quantize(l2), vmax=quantize(h2))
            elif all_non_pos:
                if l1 < 0.1 * l2 or h2 == 0:
                    norm_col = Normalize(vmin=-quantize(-l2), vmax=0)
                else:
                    norm_col = SymLogNorm(linthresh=quantize(-h2),
                                          vmin=-quantize(-l2),
                                          vmax=-quantize(-h2))
            else:
                vlim = quantize(max(-l2, h2))
                if w1 > 0.1 * w2 or l1 * h1 >= 0:
                    norm_col = Normalize(vmin=-vlim, vmax=vlim)
                else:
                    linthresh = quantize(min(-l1, h1))
                    linscale = np.log10(vlim / linthresh)
                    norm_col = SymLogNorm(linthresh=linthresh, linscale=linscale,
                                          vmin=-vlim, vmax=vlim, base=10)

        for c in range(field.shape[0]):
            s = (c,) + tuple(d // 2 for d in field.shape[1:-2])
            if size is None:
                s += (slice(None),) * 2
            else:
                s += (
                    slice(
                        (field.shape[-2] - size) // 2,
                        (field.shape[-2] + size) // 2,
                    ),
                    slice(
                        (field.shape[-1] - size) // 2,
                        (field.shape[-1] + size) // 2,
                    ),
                )

            axes[c, f].pcolormesh(field[s], cmap=cmap_col, norm=norm_col)

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


def plt_power(*fields, dis=None, label=None, **kwargs):
    """Plot power spectra of fields.

    Each field should have batch and channel dimensions followed by spatial
    dimensions.

    Optionally the field can be transformed by lag2eul first if given `dis`.

    See `map2map.models.power`.
    """
    plt.close('all')

    if label is not None:
        assert len(label) == len(fields) or len(label) == len(dis)
    else:
        label = [None] * len(fields)

    with torch.no_grad():
        if dis:
            fields = lag2eul(dis, val=fields, **kwargs)

        ks, Ps = [], []
        for field in fields:
            k, P, _ = power(field)
            ks.append(k)
            Ps.append(P)

    ks = [k.cpu().numpy() for k in ks]
    Ps = [P.cpu().numpy() for P in Ps]

    fig, axes = plt.subplots(figsize=(4.8, 3.6), dpi=150)

    for k, P, l in zip(ks, Ps, label):
        axes.loglog(k, P, label=l, alpha=0.7)

    axes.legend()
    axes.set_xlabel('unnormalized wavenumber')
    axes.set_ylabel('unnormalized power')

    fig.tight_layout()

    return fig
