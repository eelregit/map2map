from math import log2
import torch
import torch.nn as nn

from .narrow import narrow_by
from .resample import Resampler
from .style import ConvStyled3d, BatchNormStyled3d, LeakyReLUStyled
from .styled_conv import ResStyledBlock
class G(nn.Module):
    def __init__(self, in_chan, out_chan, style_size, scale_factor=16,
                 chan_base=512, chan_min=64, chan_max=512, cat_noise=False,
                 **kwargs):
        super().__init__()
        self.style_size = style_size
        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))

        assert chan_min <= chan_max

        def chan(b):
            c = chan_base >> b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.block0 = nn.Sequential(
            ConvStyled3d(in_chan, chan(0), style_size=self.style_size ,1),
            LeakyReLUStyled(0.2, True),
        )

        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            prev_chan, next_chan = chan(b), chan(b + 1)
            self.blocks.append(
                HBlock(prev_chan, next_chan, out_chan, cat_noise))

    def forward(self, x, style=s):
        y = x  # direct upsampling from the input
        x = self.block0(x, s)

        #y = None  # no direct upsampling from the input
        for block in self.blocks:
            x, y = block(x, y)

        return y





class HBlock(nn.Module):
    """The "H" block of the StyleGAN2 generator.

        x_p                     y_p
         |                       |
    convolution           linear upsample
         |                       |
          >--- projection ------>+
         |                       |
         v                       v
        x_n                     y_n

    See Fig. 7 (b) upper in https://arxiv.org/abs/1912.04958
    Upsampling are all linear, not transposed convolution.

    Parameters
    ----------
    prev_chan : number of channels of x_p
    next_chan : number of channels of x_n
    out_chan : number of channels of y_p and y_n
    cat_noise: concatenate noise if True, otherwise add noise

    Notes
    -----
    next_size = 2 * prev_size - 6
    """
    def __init__(self, prev_chan, next_chan, out_chan, cat_noise):
        super().__init__()

        self.upsample = Resampler(3, 2)

        self.conv = nn.Sequential(
            AddNoise(cat_noise, chan=prev_chan),
            self.upsample,
            nn.Conv3d(prev_chan + int(cat_noise), next_chan, 3),
            nn.LeakyReLU(0.2, True),

            AddNoise(cat_noise, chan=next_chan),
            nn.Conv3d(next_chan + int(cat_noise), next_chan, 3),
            nn.LeakyReLU(0.2, True),
        )

        self.proj = nn.Sequential(
            nn.Conv3d(next_chan, out_chan, 1),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x, y):
        x = self.conv(x)  # narrow by 3

        if y is None:
            y = self.proj(x)
        else:
            y = self.upsample(y)  # narrow by 1

            y = narrow_by(y, 2)

            y = y + self.proj(x)

        return x, y


class AddNoise(nn.Module):
    """Add or concatenate noise.

    Add noise if `cat=False`.
    The number of channels `chan` should be 1 (StyleGAN2)
    or that of the input (StyleGAN).
    """
    def __init__(self, cat, chan=1):
        super().__init__()

        self.cat = cat

        if not self.cat:
            self.std = nn.Parameter(torch.zeros([chan]))

    def forward(self, x):
        noise = torch.randn_like(x[:, :1])

        if self.cat:
            x = torch.cat([x, noise], dim=1)
        else:
            std_shape = (-1,) + (1,) * (x.dim() - 2)
            noise = self.std.view(std_shape) * noise

            x = x + noise

        return x


class D(nn.Module):
    def __init__(self, in_chan, out_chan, style_size, scale_factor=16,
                 chan_base=512, chan_min=64, chan_max=512,
                 **kwargs):
        super().__init__()

        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))
        self.style_size = style_size

        assert chan_min <= chan_max

        def chan(b):
            if b >= 0:
                c = chan_base >> b
            else:
                c = chan_base << -b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.block0 = nn.Sequential(
            ConvStyled3d(in_chan, chan(num_blocks),style_size= self.style_size ,1),
            LeakyReLUStyled(0.2, True),
        )

        self.blocks = nn.ModuleList()
        for b in reversed(range(num_blocks)):
            prev_chan, next_chan = chan(b+1), chan(b)
            self.blocks.append(ResStyledBlock(in_chan=prev_chan, out_chan=next_chan, seq='CACA', last_act=False))
            self.blocks.append(Resampler(3, 0.5))

        self.block9 = nn.Sequential(
            ConvStyled3d(chan(0), chan(-1),style_size = self.style_size, 1),
            LeakyReLUStyled(0.2, True),
            ConvStyled3d(chan(-1), 1,style_size=self.style_size, 1),
        )

    def forward(self, x, style=s):
        x = self.block0(x, s)

        for block in self.blocks:
            x = block(x, s)

        x = self.block9(x, s)

        return x


