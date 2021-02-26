import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    """Pixelwise normalization after conv layers.

    See ProGAN, StyleGAN.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, eps=1e-8):
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)


class LinearElr(nn.Module):
    """Linear layer with equalized learning rate.

    See ProGAN, StyleGAN, and 1706.05350

    Useful at all if not for regularization(1706.05350)?
    """
    def __init__(self, in_size, out_size, bias=True, act=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_size, in_size))
        self.wnorm = 1 / math.sqrt(in_size)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_size))
        else:
            self.register_parameter('bias', None)

        self.act = act

    def forward(self, x):
        x = F.linear(x, self.weight * self.wnorm, bias=self.bias)

        if self.act:
            x = F.leaky_relu(x, negative_slope=0.2)

        return x


class ConvElr3d(nn.Module):
    """Conv3d layer with equalized learning rate.

    See ProGAN, StyleGAN, and 1706.05350

    Useful at all if not for regularization(1706.05350)?
    """
    def __init__(self, in_chan, out_chan, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_chan, in_chan, *(kernel_size,) * 3),
        )
        fan_in = in_chan * kernel_size ** 3
        self.wnorm = 1 / math.sqrt(fan_in)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_chan))
        else:
            self.register_parameter('bias', None)

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = F.conv2d(
            x,
            self.weight * self.wnorm,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return x


class ConvMod3d(nn.Module):
    """Convolution layer with modulation and demodulation, from StyleGAN2.

    Weight and bias initialization from `torch.nn._ConvNd.reset_parameters()`.
    """
    def __init__(self, style_size, in_chan, out_chan, kernel_size=3, stride=1,
                 bias=True, resample=None):
        super().__init__()

        self.style_weight = nn.Parameter(torch.empty(in_chan, style_size))
        nn.init.kaiming_uniform_(self.style_weight, a=math.sqrt(5),
                                 mode='fan_in', nonlinearity='leaky_relu')
        self.style_bias = nn.Parameter(torch.ones(in_chan))  # NOTE: init to 1

        if resample is None:
            K3 = (kernel_size,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride = stride
            self.conv = F.conv3d
        elif resample == 'U':
            K3 = (2,) * 3
            # NOTE not clear to me why convtranspose have channels swapped
            self.weight = nn.Parameter(torch.empty(in_chan, out_chan, *K3))
            self.stride = 2
            self.conv = F.conv_transpose3d
        elif resample == 'D':
            K3 = (2,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride = 2
            self.conv = F.conv3d
        else:
            raise ValueError('resample type {} not supported'.format(resample))
        self.resample = resample

        nn.init.kaiming_uniform_(
            self.weight, a=math.sqrt(5),
            mode='fan_in',  # effectively 'fan_out' for 'D'
            nonlinearity='leaky_relu',
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_chan))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, s, eps=1e-8):
        N, Cin, *DHWin = x.shape
        C0, C1, *K3 = self.weight.shape
        if self.resample == 'U':
            Cin, Cout = C0, C1
        else:
            Cout, Cin = C0, C1

        s = F.linear(s, self.style_weight, bias=self.style_bias)

        # modulation
        if self.resample == 'U':
            s = s.reshape(N, Cin, 1, 1, 1, 1)
        else:
            s = s.reshape(N, 1, Cin, 1, 1, 1)
        w = self.weight * s

        # demodulation
        if self.resample == 'U':
            fan_in_dim = (1, 3, 4, 5)
        else:
            fan_in_dim = (2, 3, 4, 5)
        w = w * torch.rsqrt(w.pow(2).sum(dim=fan_in_dim, keepdim=True) + eps)

        w = w.reshape(N * C0, C1, *K3)
        x = x.reshape(1, N * Cin, *DHWin)
        x = self.conv(x, w, bias=self.bias, stride=self.stride, groups=N)
        _, _, *DHWout = x.shape
        x = x.reshape(N, Cout, *DHWout)

        return x
