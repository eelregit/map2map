import torch
import torch.nn as nn

from .resample import get_resampler
from .conv import narrow_by, narrow_like


class G1(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.upsample = get_resampler(3, 2)

        self.conv0 = nn.Sequential(
            nn.Conv3d(in_chan, 512, 5),
            nn.LeakyReLU(0.2, True),
        )

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 2, stride=2),
            AddNoise(),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(256, 256, 3),
            AddNoise(),
            nn.LeakyReLU(0.2, True),
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, stride=2),
            AddNoise(),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(128, 128, 3),
            AddNoise(),
            nn.LeakyReLU(0.2, True),
        )

        self.proj1 = nn.Sequential(
            nn.Conv3d(256, out_chan, 1),
            AddNoise(),
            nn.LeakyReLU(0.2, True),
        )

        self.proj2 = nn.Sequential(
            nn.Conv3d(128, out_chan, 1),
            AddNoise(),
            nn.LeakyReLU(0.2, True),
        )

        self.scale_factor = 4

    def forward(self, x):
        x = self.conv0(x)

        x = self.conv1(x)

        y = self.proj1(x)
        y = self.upsample(y)
        y = narrow_by(y, 1)

        x = self.conv2(x)

        x = self.proj2(x)
        x = narrow_like(x, y)
        y = y + x

        return y


class D1(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.net = nn.Sequential(
            ResBlock(in_chan, 128),
            ResBlock(128, 256),
            ResBlock(256, 512),
            nn.Conv3d(512, 1024, 1),
        )

    def forward(self, x):
        return self.net(x)


class AddNoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.std = nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        noise = self.std * torch.randn_like(x[:, 0])
        return x + noise


class ResBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_chan, in_chan, 3),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(in_chan, out_chan, 2, stride=2),
            nn.LeakyReLU(0.2, True),
        )

        self.skip = get_resampler(3, 0.5)

    def forward(self, x):
        y = self.conv(x)
        x = narrow_like(x, y)
        return x + y
