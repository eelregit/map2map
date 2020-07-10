import torch
import torch.nn as nn


class Lag2Eul(nn.Module):
    """Transform fields from Lagrangian description to Eulerian description

    Only works for 3d fields, output same mesh size as input.

    Input of shape `(N, C, ...)` is first split into `(N, 3, ...)` and
    `(N, C-3, ...)`. Take the former as the displacement field to map the
    latter from Lagrangian to Eulerian positions and then "paint" with CIC
    (trilinear) scheme. Use 1 if the latter is empty.

    Implementation follows pmesh/cic.py by Yu Feng.
    """
    def __init__(self):
        super().__init__()

        # FIXME for other redshift, box and mesh sizes
        from ..data.norms.cosmology import D
        z = 0
        Boxsize = 1000
        Nmesh = 512
        self.dis_norm = 6 * D(z) * Nmesh / Boxsize  # to mesh unit

    def forward(self, *xs, rm_dis_mean=True, periodic=False):
        if any(x.shape[1] < 3 for x in xs):
            raise ValueError('displacement not available with <3 channels')

        # common mean displacement of all inputs
        # if removed, fewer particles go outside of the box
        # common for all inputs so outputs are comparable in the same coords
        dis_mean = 0
        if rm_dis_mean:
            dis_mean = sum(x[:, :3].detach().mean((2, 3, 4), keepdim=True)
                        for x in xs) / len(xs)

        out = []
        for x in xs:
            N, Cin, DHW = x.shape[0], x.shape[1], x.shape[2:]

            if Cin == 3:
                Cout = 1
                val = 1
            else:
                Cout = Cin - 3
                val = x[:, 3:].contiguous().view(N, Cout, -1, 1)
            mesh = torch.zeros(N, Cout, *DHW, dtype=x.dtype, device=x.device)

            pos = (x[:, :3] - dis_mean) * self.dis_norm

            pos[:, 0] += torch.arange(0.5, DHW[0], device=x.device)[:, None, None]
            pos[:, 1] += torch.arange(0.5, DHW[1], device=x.device)[:, None]
            pos[:, 2] += torch.arange(0.5, DHW[2], device=x.device)

            pos = pos.contiguous().view(N, 3, -1, 1)

            intpos = pos.floor().to(torch.int)
            neighbors = (torch.arange(8, device=x.device)
                >> torch.arange(3, device=x.device)[:, None, None] ) & 1
            tgtpos = intpos + neighbors
            del intpos, neighbors

            # CIC
            kernel = (1.0 - torch.abs(pos - tgtpos)).prod(1, keepdim=True)
            del pos

            val = val * kernel
            del kernel

            tgtpos = tgtpos.view(N, 3, -1)  # fuse spatial and neighbor axes
            val = val.view(N, Cout, -1)

            for n in range(N):  # because ind has variable length
                bounds = torch.tensor(DHW, device=x.device)[:, None]

                if periodic:
                    torch.remainder(tgtpos[n], bounds, out=tgtpos[n])

                ind = (tgtpos[n, 0] * DHW[1] + tgtpos[n, 1]
                    ) * DHW[2] + tgtpos[n, 2]
                src = val[n]

                if not periodic:
                    mask = ((tgtpos[n] >= 0) & (tgtpos[n] < bounds)).all(0)
                    ind = ind[mask]
                    src = src[:, mask]

                mesh[n].view(Cout, -1).index_add_(1, ind, src)

            out.append(mesh)

        return out
