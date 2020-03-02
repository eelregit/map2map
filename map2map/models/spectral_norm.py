import torch.nn as nn
from torch.nn.utils import spectral_norm, remove_spectral_norm


def add_spectral_norm(module):
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            setattr(module, name, spectral_norm(child))
        else:
            add_spectral_norm(child)


def rm_spectral_norm(module):
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            setattr(module, name, remove_spectral_norm(child))
        else:
            rm_spectral_norm(child)
