from math import log2
import torch
import torch.nn as nn

from map2map.models.narrow import narrow_by, narrow_like
from map2map.models.resample import Resampler


class G2(nn.Module):
    def __init__(self, in_chan, out_chan, scale_factor=8,
                 chan_base=512, chan_min=64, chan_max=512, cat_noise=False):
        super().__init__()

        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))

        assert chan_min <= chan_max

        def chan(b):
            c = chan_base >> b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.block0 = nn.Sequential(
            nn.Conv2d(in_chan, chan(0), 1),
            nn.LeakyReLU(0.2, True),
        )

        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            prev_chan, next_chan = chan(b), chan(b+1)
            self.blocks.append(
                SkipBlock(prev_chan, next_chan, out_chan, cat_noise))

    def forward(self, x):
        x = self.block0(x)
        x, y = self.blocks[0](x, None)
        y = y * 0
        for block in self.blocks[1:  ]:
            x, y = block(x, y)
        return y


class SkipBlock(nn.Module):
    """The "I" block of the StyleGAN2 generator.

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

        self.upsample = Resampler(2, 2, narrow=False)

        self.conv = nn.Sequential(
            AddNoise(cat_noise, chan=prev_chan),
            self.upsample,
            nn.Conv2d(prev_chan + int(cat_noise), next_chan, 3, padding = 1),
            nn.LeakyReLU(0.2, True),

            AddNoise(cat_noise, chan=next_chan),
            nn.Conv2d(next_chan + int(cat_noise), next_chan, 3, padding = 1),
            nn.LeakyReLU(0.2, True),
        )

        self.proj = nn.Sequential(
            nn.Conv2d(next_chan, out_chan, 1),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x, y):
        x = self.conv(x)  # narrow by 3

        if y is None:
            y = self.proj(x)
        else:
            y = self.upsample(y)  # narrow by 1

            #y = narrow_by(y, 2)

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

        return x + noise


class D2(nn.Module):
    def __init__(self, in_chan, out_chan, scale_factor=8,
                 chan_base=512, chan_min=64, chan_max=512):
        super().__init__()

        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))

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
            nn.Conv2d(in_chan, chan(num_blocks), 1),
            nn.LeakyReLU(0.2, True),
        )

        self.blocks = nn.ModuleList()
        for b in reversed(range(num_blocks)):
            prev_chan, next_chan = chan(b+1), chan(b)
            self.blocks.append(ResBlock(prev_chan, next_chan))

        self.block9 = nn.Sequential(
            #input shape is chan(0) x in_size/scale_factor x in_size/scale_factor
            #with current config that is 512 x 64 x 64
            nn.Conv2d(chan(0), chan(0), kernel_size=4, stride = 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            #input shape =  chan(0) x 32 x 32
            nn.Conv2d(chan(0), chan(0), kernel_size=4, stride = 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            #input shape =  chan(0) x 16 x 16
            nn.Conv2d(chan(0), chan(0), kernel_size=4, stride = 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            #input shape =  chan(0) x 8 x 8
            nn.Conv2d(chan(0), chan(0), kernel_size=4, stride = 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            #input shape =  chan(0) x 4 x 4
            nn.Conv2d(chan(0), 1, kernel_size=4, stride = 1, padding=0),
            #output shape = 1x1x1
            
            )

    def forward(self, x):
        x = self.block0(x)

        for block in self.blocks:
            x = block(x)

        x = self.block9(x)

        return x


class ResBlock(nn.Module):
    """The residual block of the StyleGAN2 discriminator.

    Downsampling are all linear, not strided convolution.

    Notes
    -----
    next_size = (prev_size - 4) // 2
    """
    def __init__(self, prev_chan, next_chan):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(prev_chan, prev_chan, 3, padding = 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(prev_chan, next_chan, 3, padding = 1),
            nn.LeakyReLU(0.2, True),
        )

        self.skip = nn.Conv2d(prev_chan, next_chan, 1)

        self.downsample = Resampler(2, 0.5, narrow=False)

    def forward(self, x):
        y = self.conv(x)

        x = self.skip(x)
        #x = narrow_by(x, 2)

        x = x + y

        x = self.downsample(x)

        return x
