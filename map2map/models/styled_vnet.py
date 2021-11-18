import torch
import torch.nn as nn

from .styled_conv import ConvStyledBlock, ResStyledBlock
from .narrow import narrow_by


class StyledVNet(nn.Module):
    def __init__(self, in_chan, out_chan, style_size, bypass=None, **kwargs):
        """V-Net like network with styles

        See `vnet.VNet`.
        """
        super().__init__()

        # activate non-identity skip connection in residual block
        # by explicitly setting out_chan
        self.conv_l0 = ResStyledBlock(in_chan, 64, style_size=style_size, seq='CACBA')
        self.down_l0 = ConvStyledBlock(64, style_size=style_size, seq='DBA')
        self.conv_l1 = ResStyledBlock(64, 64, style_size=style_size, seq='CBACBA')
        self.down_l1 = ConvStyledBlock(64, style_size=style_size, seq='DBA')

        self.conv_c = ResStyledBlock(64, 64, style_size=style_size, seq='CBACBA')

        self.up_r1 = ConvStyledBlock(64, style_size=style_size, seq='UBA')
        self.conv_r1 = ResStyledBlock(128, 64, style_size=style_size, seq='CBACBA')
        self.up_r0 = ConvStyledBlock(64, style_size=style_size, seq='UBA')
        self.conv_r0 = ResStyledBlock(128, out_chan, style_size=style_size, seq='CAC')

        if bypass is None:
            self.bypass = in_chan == out_chan
        else:
            self.bypass = bypass

    def forward(self, x, style=s):
        if self.bypass:
            x0 = x

        y0 = self.conv_l0(x, s)
        x = self.down_l0(y0, s)

        y1 = self.conv_l1(x, s)
        x = self.down_l1(y1, s)

        x = self.conv_c(x, s)

        x = self.up_r1(x, s)
        y1 = narrow_by(y1, 4)
        x = torch.cat([y1, x], dim=1)
        del y1
        x = self.conv_r1(x, s)

        x = self.up_r0(x, s)
        y0 = narrow_by(y0, 16)
        x = torch.cat([y0, x], dim=1)
        del y0
        x = self.conv_r0(x, s)

        if self.bypass:
            x0 = narrow_by(x0, 20)
            x += x0

        return x
