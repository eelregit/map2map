import torch
import torch.nn as nn

from .swish import Swish


class ConvBlock(nn.Module):
    """Convolution blocks of the form specified by `seq`.

    `seq` types:
    'C': convolution specified by `kernel_size` and `stride`
    'B': normalization (to be renamed to 'N')
    'A': activation
    'U': upsampling transposed convolution of kernel size 2 and stride 2
    'D': downsampling convolution of kernel size 2 and stride 2
    """
    def __init__(self, in_chan, out_chan=None, mid_chan=None,
            kernel_size=3, stride=1, seq='CBA'):
        super().__init__()

        if out_chan is None:
            out_chan = in_chan

        self.in_chan = in_chan
        self.out_chan = out_chan
        if mid_chan is None:
            self.mid_chan = max(in_chan, out_chan)
        self.kernel_size = kernel_size
        self.stride = stride

        self.norm_chan = in_chan
        self.idx_conv = 0
        self.num_conv = sum([seq.count(l) for l in ['U', 'D', 'C']])

        layers = [self._get_layer(l) for l in seq]

        self.convs = nn.Sequential(*layers)

    def _get_layer(self, l):
        if l == 'U':
            in_chan, out_chan = self._setup_conv()
            return nn.ConvTranspose3d(in_chan, out_chan, 2, stride=2)
        elif l == 'D':
            in_chan, out_chan = self._setup_conv()
            return nn.Conv3d(in_chan, out_chan, 2, stride=2)
        elif l == 'C':
            in_chan, out_chan = self._setup_conv()
            return nn.Conv3d(in_chan, out_chan, self.kernel_size,
                    stride=self.stride)
        elif l == 'B':
            return nn.BatchNorm3d(self.norm_chan)
            #return nn.InstanceNorm3d(self.norm_chan, affine=True, track_running_stats=True)
            #return nn.InstanceNorm3d(self.norm_chan)
        elif l == 'A':
            return nn.LeakyReLU()
        else:
            raise NotImplementedError('layer type {} not supported'.format(l))

    def _setup_conv(self):
        self.idx_conv += 1

        in_chan = out_chan = self.mid_chan
        if self.idx_conv == 1:
            in_chan = self.in_chan
        if self.idx_conv == self.num_conv:
            out_chan = self.out_chan

        self.norm_chan = out_chan

        return in_chan, out_chan

    def forward(self, x):
        return self.convs(x)


class ResBlock(ConvBlock):
    """Residual convolution blocks of the form specified by `seq`. Input is added
    to the residual followed by an optional activation (trailing `'A'` in `seq`).

    See `ConvBlock` for `seq` types.
    """
    def __init__(self, in_chan, out_chan=None, mid_chan=None,
            seq='CBACBA'):
        super().__init__(in_chan, out_chan=out_chan,
                mid_chan=mid_chan,
                seq=seq[:-1] if seq[-1] == 'A' else seq)

        if out_chan is None:
            self.skip = None
        else:
            self.skip = nn.Conv3d(in_chan, out_chan, 1)

        if 'U' in seq or 'D' in seq:
            raise NotImplementedError('upsample and downsample layers '
                    'not supported yet')

        if seq[-1] == 'A':
            self.act = nn.LeakyReLU()
        else:
            self.act = None

    def forward(self, x):
        y = x

        if self.skip is not None:
            y = self.skip(y)

        x = self.convs(x)

        y = narrow_like(y, x)
        x += y

        if self.act is not None:
            x = self.act(x)

        return x


def narrow_like(a, b):
    """Narrow a to be like b.

    Try to be symmetric but cut more on the right for odd difference,
    consistent with the downsampling.
    """
    for d in range(2, a.dim()):
        width = int(a.shape[d]) - int(b.shape[d])
        half_width = int(width // 2)
        a = a.narrow(d, half_width, int(a.shape[d]) - width)
    return a

