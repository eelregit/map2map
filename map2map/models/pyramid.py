import torch
import torch.nn as nn

from .conv import ConvBlock, ResBlock, narrow_like


class PyramidNet(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.down = nn.AvgPool3d(2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_l0 = ResBlock(in_chan, 64, seq="CAC")
        self.conv_l1 = ResBlock(64, seq="CBAC")

        self.conv_c = ResBlock(64, seq="CBAC")

        self.conv_r1 = ResBlock(128, 64, seq="CBAC")
        self.conv_r0 = ResBlock(128, out_chan, seq="CAC")

    def forward(self, x):
        y0 = self.conv_l0(x)
        x = self.down(y0)
        y0 = y0 - self.up(x)

        y1 = self.conv_l1(x)
        x = self.down(y1)
        y1 = y1 - self.up(x)

        x = self.conv_c(x)

        x = self.up(x)
        y1 = narrow_like(y1, x)
        x = torch.cat([y1, x], dim=1)
        del y1
        x = self.conv_r1(x)

        x = self.up(x)
        y0 = narrow_like(y0, x)
        x = torch.cat([y0, x], dim=1)
        del y0
        x = self.conv_r0(x)

        return x
