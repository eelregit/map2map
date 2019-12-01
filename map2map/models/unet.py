import torch
import torch.nn as nn

from .conv import ConvBlock, ResBlock, narrow_like


class DownBlock(ConvBlock):
    def __init__(self, in_channels, out_channels, seq='BADBA'):
        super().__init__(in_channels, out_channels, seq=seq)

class UpBlock(ConvBlock):
    def __init__(self, in_channels, out_channels, seq='BAUBA'):
        super().__init__(in_channels, out_channels, seq=seq)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_0l = nn.Sequential(
            ConvBlock(in_channels, 64, seq='CA'),
            ResBlock(64, seq='CBACBACB'),
        )
        self.down_0l = ConvBlock(64, 128, seq='DBA')
        self.conv_1l = nn.Sequential(
            ResBlock(128, seq='CBACB'),
            ResBlock(128, seq='CBACB'),
        )
        self.down_1l = ConvBlock(128, 256, seq='DBA')

        self.conv_2c = nn.Sequential(
            ResBlock(256, seq='CBACB'),
            ResBlock(256, seq='CBACB'),
        )

        self.up_1r = ConvBlock(256, 128, seq='UBA')
        self.conv_1r = nn.Sequential(
            ResBlock(256, seq='CBACB'),
            ResBlock(256, seq='CBACB'),
        )
        self.up_0r = ConvBlock(256, 64, seq='UBA')
        self.conv_0r = nn.Sequential(
            ResBlock(128, seq='CBACBAC'),
            ConvBlock(128, out_channels, seq='C')
        )

    def forward(self, x):
        y0 = self.conv_0l(x)
        x = self.down_0l(y0)

        y1 = self.conv_1l(x)
        x = self.down_1l(y1)

        x = self.conv_2c(x)

        x = self.up_1r(x)
        y1 = narrow_like(y1, x)
        x = torch.cat([y1, x], dim=1)
        del y1
        x = self.conv_1r(x)

        x = self.up_0r(x)
        y0 = narrow_like(y0, x)
        x = torch.cat([y0, x], dim=1)
        del y0
        x = self.conv_0r(x)

        return x
