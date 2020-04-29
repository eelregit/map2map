import torch
import torch.nn as nn


def narrow_by(a, c):
    """Narrow a by size c symmetrically on all sides
    """
    assert isinstance(c, int)

    for d in range(2, a.dim()):
        a = a.narrow(d, c, a.shape[d] - 2 * c)
    return a


def narrow_like(a, b):
    """Narrow a to be like b.

    Try to be symmetric but cut more on the right for odd difference
    """
    for d in range(2, a.dim()):
        width = a.shape[d] - b.shape[d]
        half_width = width // 2
        a = a.narrow(d, half_width, a.shape[d] - width)
    return a
