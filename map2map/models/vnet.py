import torch
import torch.nn as nn

from .conv import ConvBlock, ResBlock, narrow_like


class VNet(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.conv_l0 = ResBlock(in_chan, 64, seq="CAC")
        self.down_l0 = ConvBlock(64, seq="BADBA")
        self.conv_l1 = ResBlock(64, seq="CBAC")
        self.down_l1 = ConvBlock(64, seq="BADBA")

        self.conv_c = ResBlock(64, seq="CBAC")

        self.up_r1 = ConvBlock(64, seq="BAUBA")
        self.conv_r1 = ResBlock(128, 64, seq="CBAC")
        self.up_r0 = ConvBlock(64, seq="BAUBA")
        self.conv_r0 = ResBlock(128, out_chan, seq="CAC")

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


class VNetFat(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.conv_l0 = nn.Sequential(
            ResBlock(in_chan, 64, seq="CACBA"), ResBlock(64, seq="CBACBA"),
        )
        self.down_l0 = ConvBlock(64, seq="DBA")
        self.conv_l1 = nn.Sequential(
            ResBlock(64, seq="CBACBA"), ResBlock(64, seq="CBACBA"),
        )  # FIXME: test CBACBA+DBA vs CBAC+BADBA
        self.down_l1 = ConvBlock(64, seq="DBA")

        self.conv_c = nn.Sequential(
            ResBlock(64, seq="CBACBA"), ResBlock(64, seq="CBACBA"),
        )

        self.up_r1 = ConvBlock(64, seq="UBA")
        self.conv_r1 = nn.Sequential(
            ResBlock(128, seq="CBACBA"), ResBlock(128, seq="CBACBA"),
        )
        self.up_r0 = ConvBlock(128, 64, seq="UBA")
        self.conv_r0 = nn.Sequential(
            ResBlock(128, seq="CBACBA"), ResBlock(128, out_chan, seq="CAC"),
        )

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
