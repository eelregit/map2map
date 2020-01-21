import torch
import torch.nn as nn

from .conv import ConvBlock, narrow_like


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_l0 = ConvBlock(in_channels, 64, seq='CAC')
        self.down_l0 = ConvBlock(64, seq='BADBA')
        self.conv_l1 = ConvBlock(64, seq='CBAC')
        self.down_l1 = ConvBlock(64, seq='BADBA')

        self.conv_c = ConvBlock(64, seq='CBAC')

        self.up_r1 = ConvBlock(64, seq='BAUBA')
        self.conv_r1 = ConvBlock(128, 64, seq='CBAC')
        self.up_r0 = ConvBlock(64, seq='BAUBA')
        self.conv_r0 = ConvBlock(128, out_channels, seq='CAC')

    def forward(self, x):
        y0 = self.conv_l0(x)
        x = self.down_l0(y0)

        y1 = self.conv_l1(x)
        x = self.down_l1(y1)

        x = self.conv_c(x)

        x = self.up_r1(x)
        y1 = narrow_like(y1, x)
        x = torch.cat([y1, x], dim=1)
        del y1
        x = self.conv_r1(x)

        x = self.up_r0(x)
        y0 = narrow_like(y0, x)
        x = torch.cat([y0, x], dim=1)
        del y0
        x = self.conv_r0(x)

        return x
