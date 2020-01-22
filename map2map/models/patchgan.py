import torch.nn as nn

from .conv import ConvBlock


class PatchGAN(nn.Module):
    def __init__(self, in_chan, out_chan=1):
        super().__init__()

        self.convs = nn.Sequential(
            ConvBlock(in_chan, 64, seq='CA'),
            ConvBlock(64, 128, seq='CBA'),
            ConvBlock(128, 256, seq='CBA'),
            nn.Conv3d(256, out_chan, 1)
        )

    def forward(self, x):
        return self.convs(x)
