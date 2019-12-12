import torch


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
