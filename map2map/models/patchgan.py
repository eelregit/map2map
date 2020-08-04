import torch.nn as nn

from .conv import ConvBlock


class PatchGAN(nn.Module):
    def __init__(self, in_chan, out_chan=1, **kwargs):
        super().__init__()

        self.convs = nn.Sequential(
            ConvBlock(in_chan, 32, seq='CA'),
            ConvBlock(32, 64, seq='CBA'),
            ConvBlock(64, 128, seq='CBA'),
            ConvBlock(128, out_chan, seq='C'),
        )

    def forward(self, x):
        return self.convs(x)


class PatchGAN42(nn.Module):
    """PatchGAN similar to the one in pix2pix
    """
    def __init__(self, in_chan, out_chan=1, **kwargs):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv3d(in_chan, 64, 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, 4, stride=2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, 4, stride=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, out_chan, 1),
        )

    def forward(self, x):
        return self.convs(x)
