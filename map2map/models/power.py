import torch

from .lag2eul import lag2eul


def power(x):
    """Compute power spectra of input fields

    Each field should have batch and channel dimensions followed by spatial
    dimensions. Powers are summed over channels, and averaged over batches.

    Power is not normalized. Wavevectors are in unit of the fundamental
    frequency of the input.
    """
    signal_ndim = x.dim() - 2
    kmax = min(d for d in x.shape[-signal_ndim:]) // 2
    even = x.shape[-1] % 2 == 0

    x = torch.rfft(x, signal_ndim)
    P = x.pow(2).sum(dim=-1)
    del x

    batch_ndim = P.dim() - signal_ndim - 1
    if batch_ndim > 0:
        P = P.mean(tuple(range(batch_ndim)))
    if P.dim() > signal_ndim:
        P = P.sum(dim=0)
    P = P.flatten()

    k = [torch.arange(d, dtype=torch.float32, device=P.device)
         for d in P.shape]
    k = torch.meshgrid(*k)
    k = torch.stack(k, dim=0)
    k = k.norm(p=2, dim=0)
    k = k.flatten()

    N = torch.full_like(P, 2, dtype=torch.int32)
    N[..., 0] = 1
    if even:
        N[..., -1] = 1
    N = N.flatten()

    kbin = k.ceil().to(torch.int32)
    k = torch.bincount(kbin, weights=k * N)
    P = torch.bincount(kbin, weights=P * N)
    N = torch.bincount(kbin, weights=N)
    del kbin

    # drop k=0 mode and cut at kmax (smallest Nyquist)
    k = k[1:1+kmax]
    P = P[1:1+kmax]
    N = N[1:1+kmax]

    k /= N
    P /= N

    return k, P, N
