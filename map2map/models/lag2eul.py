import itertools
import torch

from ..data.norms.cosmology import D


def lag2eul(
        dis,
        val=1.0,
        eul_scale_factor=1,
        eul_pad=0,
        rm_dis_mean=True,
        periodic=False,
        z=0.0,
        dis_std=6.0,
        boxsize=1000.,
        meshsize=512,
        **kwargs):
    """Transform fields from Lagrangian description to Eulerian description

    Only works for 3d fields, output same mesh size as input.

    Use displacement fields `dis` to map the value fields `val` from Lagrangian
    to Eulerian positions and then "paint" with CIC (trilinear) scheme.
    Displacement and value fields are paired when are sequences of the same
    length. If the displacement (value) field has only one entry, it is shared
    by all the value (displacement) fields.

    The Eulerian size is scaled by the `eul_scale_factor` and then padded by
    the `eul_pad`.

    Common mean displacement of all inputs can be removed to bring more
    particles inside the box. Periodic boundary condition can be turned on.

    Note that the box and mesh sizes don't have to be that of the inputs, as
    long as their ratio gives the right resolution. One can therefore set them
    to the values of the whole Lagrangian fields, and use smaller inputs.

    Implementation follows pmesh/cic.py by Yu Feng.
    """
    # NOTE the following factor assumes normalized displacements
    # and thus undoes it
    dis_norm = dis_std * D(z) * meshsize / boxsize  # to mesh unit
    dis_norm *= eul_scale_factor

    if isinstance(dis, torch.Tensor):
        dis = [dis]
    if isinstance(val, (float, torch.Tensor)):
        val = [val]
    if len(dis) != len(val) and len(dis) != 1 and len(val) != 1:
        raise ValueError('dis-val field mismatch')

    if any(d.dim() != 5 for d in dis):
        raise NotImplementedError('only support 3d fields for now')
    if any(d.shape[1] != 3 for d in dis):
        raise ValueError('only support 3d displacement fields')

    # common mean displacement of all inputs
    # if removed, fewer particles go outside of the box
    # common for all inputs so outputs are comparable in the same coords
    d_mean = 0
    if rm_dis_mean:
        d_mean = sum(d.detach().mean((2, 3, 4), keepdim=True)
                     for d in dis) / len(dis)

    out = []
    if len(dis) == 1 and len(val) != 1:
        dis = itertools.repeat(dis[0])
    elif len(dis) != 1 and len(val) == 1:
        val = itertools.repeat(val[0])
    for d, v in zip(dis, val):
        dtype, device = d.dtype, d.device

        N, DHW = d.shape[0], d.shape[2:]
        DHW = torch.Size([s * eul_scale_factor + 2 * eul_pad for s in DHW])

        if isinstance(v, float):
            C = 1
        else:
            C = v.shape[1]
            v = v.contiguous().flatten(start_dim=2).unsqueeze(-1)

        mesh = torch.zeros(N, C, *DHW, dtype=dtype, device=device)

        pos = (d - d_mean) * dis_norm
        del d

        pos[:, 0] += torch.arange(0, DHW[0] - 2 * eul_pad, eul_scale_factor,
                                  dtype=dtype, device=device)[:, None, None]
        pos[:, 1] += torch.arange(0, DHW[1] - 2 * eul_pad, eul_scale_factor,
                                  dtype=dtype, device=device)[:, None]
        pos[:, 2] += torch.arange(0, DHW[2] - 2 * eul_pad, eul_scale_factor,
                                  dtype=dtype, device=device)

        pos = pos.contiguous().view(N, 3, -1, 1)  # last axis for neighbors

        intpos = pos.floor().to(torch.int)
        neighbors = (
            torch.arange(8, device=device)
            >> torch.arange(3, device=device)[:, None, None]
        ) & 1
        tgtpos = intpos + neighbors
        del intpos, neighbors

        # CIC
        kernel = (1.0 - torch.abs(pos - tgtpos)).prod(1, keepdim=True)
        del pos

        v = v * kernel
        del kernel

        tgtpos = tgtpos.view(N, 3, -1)  # fuse spatial and neighbor axes
        v = v.view(N, C, -1)

        for n in range(N):  # because ind has variable length
            bounds = torch.tensor(DHW, device=device)[:, None]

            if periodic:
                torch.remainder(tgtpos[n], bounds, out=tgtpos[n])

            ind = (tgtpos[n, 0] * DHW[1] + tgtpos[n, 1]
                   ) * DHW[2] + tgtpos[n, 2]
            src = v[n]

            if not periodic:
                mask = ((tgtpos[n] >= 0) & (tgtpos[n] < bounds)).all(0)
                ind = ind[mask]
                src = src[:, mask]

            mesh[n].view(C, -1).index_add_(1, ind, src)

        out.append(mesh)

    return out
