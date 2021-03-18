import warnings
import torch
import torch.nn as nn

from .narrow import narrow_like
from .style import ConvStyled3d, BatchNormStyled3d, LeakyReLUStyled


class ConvStyledBlock(nn.Module):
    """Convolution blocks of the form specified by `seq`.

    `seq` types:
    'C': convolution specified by `kernel_size` and `stride`
    'B': normalization (to be renamed to 'N')
    'A': activation
    'U': upsampling transposed convolution of kernel size 2 and stride 2
    'D': downsampling convolution of kernel size 2 and stride 2
    """
    def __init__(self, style_size, in_chan, out_chan=None, mid_chan=None,
            kernel_size=3, stride=1, seq='CBA'):
        super().__init__()

        if out_chan is None:
            out_chan = in_chan

        self.style_size = style_size
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

        self.convs = nn.ModuleList(layers)

    def _get_layer(self, l):
        if l == 'U':
            in_chan, out_chan = self._setup_conv()
            return ConvStyled3d(self.style_size, in_chan, out_chan, 2, stride=2,
                                resample = 'U')
        elif l == 'D':
            in_chan, out_chan = self._setup_conv()
            return ConvStyled3d(self.style_size, in_chan, out_chan, 2, stride=2,
                                resample = 'D')
        elif l == 'C':
            in_chan, out_chan = self._setup_conv()
            return ConvStyled3d(self.style_size, in_chan, out_chan, self.kernel_size,
                                stride=self.stride)
        elif l == 'B':
            return BatchNormStyled3d(self.norm_chan)
        elif l == 'A':
            return LeakyReLUStyled()
        else:
            raise ValueError('layer type {} not supported'.format(l))

    def _setup_conv(self):
        self.idx_conv += 1

        in_chan = out_chan = self.mid_chan
        if self.idx_conv == 1:
            in_chan = self.in_chan
        if self.idx_conv == self.num_conv:
            out_chan = self.out_chan

        self.norm_chan = out_chan

        return in_chan, out_chan

    def forward(self, x, s):
        for l in self.convs:
            x = l(x, s)
        return x


class ResStyledBlock(ConvStyledBlock):
    """Residual convolution blocks of the form specified by `seq`.
    Input, via a skip connection, is added to the residual followed by an
    optional activation.

    The skip connection is identity if `out_chan` is omitted, otherwise it uses
    a size 1 "convolution", i.e. one can trigger the latter by setting
    `out_chan` even if it equals `in_chan`.

    A trailing `'A'` in seq can either operate before or after the addition,
    depending on the boolean value of `last_act`, defaulting to `seq[-1] == 'A'`

    See `ConvStyledBlock` for `seq` types.
    """
    def __init__(self, style_size, in_chan, out_chan=None, mid_chan=None,
                 kernel_size=3, stride=1, seq='CBACBA', last_act=None):
        if last_act is None:
            last_act = seq[-1] == 'A'
        elif last_act and seq[-1] != 'A':
            warnings.warn(
                'Disabling last_act without trailing activation in seq',
                RuntimeWarning,
            )
            last_act = False

        if last_act:
            seq = seq[:-1]

        super().__init__(style_size, in_chan, out_chan=out_chan, mid_chan=mid_chan,
                         kernel_size=kernel_size, stride=stride, seq=seq)

        if last_act:
            self.act = LeakyReLUStyled()
        else:
            self.act = None

        if out_chan is None:
            self.skip = None
        else:
            self.skip = ConvStyled3d(style_size, in_chan, out_chan, 1)

        if 'U' in seq or 'D' in seq:
            raise NotImplementedError('upsample and downsample layers '
                    'not supported yet')

    def forward(self, x, s):
        y = x

        if self.skip is not None:
            y = self.skip(y, s)

        for l in self.convs:
            x = l(x, s)

        y = narrow_like(y, x)
        x += y

        if self.act is not None:
            x = self.act(x, s)

        return x
