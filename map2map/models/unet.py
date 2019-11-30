import torch
import torch.nn as nn

from .conv import ConvBlock, narrow_like


class DownBlock(ConvBlock):
    def __init__(self, in_channels, out_channels, seq='BADBA'):
        super().__init__(in_channels, out_channels, seq=seq)

class UpBlock(ConvBlock):
    def __init__(self, in_channels, out_channels, seq='BAUBA'):
        super().__init__(in_channels, out_channels, seq=seq)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_0l = ConvBlock(in_channels, 64, seq='CAC')
        self.down_0l = DownBlock(64, 64)
        self.conv_1l = ConvBlock(64, 64)
        self.down_1l = DownBlock(64, 64)

        self.conv_2c = ConvBlock(64, 64)

        self.up_1r = UpBlock(64, 64)
        self.conv_1r = ConvBlock(128, 64)
        self.up_0r = UpBlock(64, 64)
        self.conv_0r = ConvBlock(128, out_channels, seq='CAC')

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
