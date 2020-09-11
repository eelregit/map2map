import torch
import torch.nn as nn

from .conv import ConvBlock
from .narrow import narrow_by


class UNet(nn.Module):
    def __init__(self, in_chan, out_chan, bypass=None, **kwargs):
        """U-Net like network

        Note:

        Global bypass connection adding the input to the output (similar to
        COLA for displacement input and output) from Alvaro Sanchez Gonzalez.
        Enabled by default when in_chan equals out_chan

        Global bypass, under additive symmetry, effectively obviates --aug-add
        """
        super().__init__()

        self.conv_l0 = ConvBlock(in_chan, 64, seq='CACBA')
        self.down_l0 = ConvBlock(64, seq='DBA')
        self.conv_l1 = ConvBlock(64, seq='CBACBA')
        self.down_l1 = ConvBlock(64, seq='DBA')

        self.conv_c = ConvBlock(64, seq='CBACBA')

        self.up_r1 = ConvBlock(64, seq='UBA')
        self.conv_r1 = ConvBlock(128, 64, seq='CBACBA')
        self.up_r0 = ConvBlock(64, seq='UBA')
        self.conv_r0 = ConvBlock(128, out_chan, seq='CAC')

        self.bypass = in_chan == out_chan

    def forward(self, x):
        if self.bypass:
            x0 = x

        y0 = self.conv_l0(x)
        x = self.down_l0(y0)

        y1 = self.conv_l1(x)
        x = self.down_l1(y1)

        x = self.conv_c(x)

        x = self.up_r1(x)
        y1 = narrow_by(y1, 4)
        x = torch.cat([y1, x], dim=1)
        del y1
        x = self.conv_r1(x)

        x = self.up_r0(x)
        y0 = narrow_by(y0, 16)
        x = torch.cat([y0, x], dim=1)
        del y0
        x = self.conv_r0(x)

        if self.bypass:
            x0 = narrow_by(x0, 20)
            x += x0

        return x
