import warnings
import torch
import torch.nn as nn

from .narrow import narrow_like


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

    def forward(self, x):
        return self.convs(x)


class ResBlock(ConvBlock):
    """Residual convolution blocks of the form specified by `seq`.
    Input, via a skip connection, is added to the residual followed by an
    optional activation.

    The skip connection is identity if `out_chan` is omitted, otherwise it uses
    a size 1 "convolution", i.e. one can trigger the latter by setting
    `out_chan` even if it equals `in_chan`.

    A trailing `'A'` in seq can either operate before or after the addition,
    depending on the boolean value of `last_act`, defaulting to `seq[-1] == 'A'`

    See `ConvBlock` for `seq` types.
    """
    def __init__(self, in_chan, out_chan=None, mid_chan=None,
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

        super().__init__(in_chan, out_chan=out_chan, mid_chan=mid_chan,
                         kernel_size=kernel_size, stride=stride, seq=seq)

        if last_act:
            self.act = nn.LeakyReLU()
        else:
            self.act = None

        if out_chan is None:
            self.skip = None
        else:
            self.skip = nn.Conv3d(in_chan, out_chan, 1)

        if 'U' in seq or 'D' in seq:
            raise NotImplementedError('upsample and downsample layers '
                    'not supported yet')

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
