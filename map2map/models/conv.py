import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolution blocks of the form specified by `seq`.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, seq='CBA'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if mid_channels is None:
            self.mid_channels = max(in_channels, out_channels)

        self.bn_channels = in_channels
        self.idx_conv = 0
        self.num_conv = sum([seq.count(l) for l in ['U', 'D', 'C']])

        layers = [self._get_layer(l) for l in seq]

        self.convs = nn.Sequential(*layers)

    def _get_layer(self, l):
        if l == 'U':
            in_channels, out_channels = self._setup_conv()
            return nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        elif l == 'D':
            in_channels, out_channels = self._setup_conv()
            return nn.Conv3d(in_channels, out_channels, 2, stride=2)
        elif l == 'C':
            in_channels, out_channels = self._setup_conv()
            return nn.Conv3d(in_channels, out_channels, 3)
        elif l == 'B':
            return nn.BatchNorm3d(self.bn_channels)
        elif l == 'A':
            return nn.PReLU()
        else:
            raise NotImplementedError('layer type {} not supported'.format(l))

    def _setup_conv(self):
        self.idx_conv += 1

        in_channels = out_channels = self.mid_channels
        if self.idx_conv == 1:
            in_channels = self.in_channels
        if self.idx_conv == self.num_conv:
            out_channels = self.out_channels

        self.bn_channels = out_channels

        return in_channels, out_channels

    def forward(self, x):
        return self.convs(x)


class ResBlock(ConvBlock):
    """Residual convolution blocks of the form specified by `seq`. Input is
    added to the residual followed by an activation.
    """
    def __init__(self, in_channels, out_channels=None, mid_channels=None,
            seq='CBACB'):
        if 'U' in seq or 'D' in seq:
            raise NotImplementedError('upsample and downsample layers '
                    'not supported yet')

        if out_channels is None:
            out_channels = in_channels
            self.skip = None
        else:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)

        super().__init__(in_channels, out_channels, mid_channels=mid_channels,
                seq=seq)

        self.act = nn.PReLU()

    def forward(self, x):
        y = x

        if self.skip is not None:
            y = self.skip(y)

        x = self.convs(x)

        y = narrow_like(y, x)
        x += y

        return self.act(x)


def narrow_like(a, b):
    """Narrow a to be like b.

    Try to be symmetric but cut more on the right for odd difference,
    consistent with the downsampling.
    """
    for dim in range(2, 5):
        width = a.size(dim) - b.size(dim)
        half_width = width // 2
        a = a.narrow(dim, half_width, a.size(dim) - width)
    return a
