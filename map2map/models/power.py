import torch


def power(x):
    """Compute power spectra of input fields

    Each field should have batch and channel dimensions followed by spatial
    dimensions. Powers are summed over channels, and averaged over batches.

    Power is not normalized. Wavevectors are in unit of the fundamental
    frequency of the input.
    """
    signal_ndim = x.dim() - 2
    signal_size = x.shape[-signal_ndim:]
    kmax = min(s for s in signal_size) // 2
    even = x.shape[-1] % 2 == 0

    try:
        x = torch.fft.rfftn(x, s=signal_size)  # new version broke BC
        P = x.real.square() + x.imag.square()
    except AttributeError:
        x = torch.rfft(x, signal_ndim)
        P = x.square().sum(dim=-1)

    P = P.mean(dim=0)
    P = P.sum(dim=0)
    del x

    k = [torch.arange(d, dtype=torch.float32, device=P.device)
         for d in P.shape]
    k = [j - len(j) * (j > len(j) // 2) for j in k[:-1]] + [k[-1]]
    k = torch.meshgrid(*k)
    k = torch.stack(k, dim=0)
    k = k.norm(p=2, dim=0)

    N = torch.full_like(P, 2, dtype=torch.int32)
    N[..., 0] = 1
    if even:
        N[..., -1] = 1

    k = k.flatten()
    P = P.flatten()
    N = N.flatten()

    kbin = k.ceil().to(torch.int32)
    k = torch.bincount(kbin, weights=k * N)
    P = torch.bincount(kbin, weights=P * N)
    N = torch.bincount(kbin, weights=N).round().to(torch.int32)
    del kbin

    # drop k=0 mode and cut at kmax (smallest Nyquist)
    k = k[1:1+kmax]
    P = P[1:1+kmax]
    N = N[1:1+kmax]

    k /= N
    P /= N

    return k, P, N
