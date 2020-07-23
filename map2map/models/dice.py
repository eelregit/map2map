import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps=0.):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        return dice_loss(input, target, self.eps)


def dice_loss(input, target, eps=0.):
    input = input.view(-1)
    target = target.view(-1)

    prod = (input * target).sum()
    in_sq = (input * input).sum()
    tgt_sq = (target * target).sum()

    dice = (2 * prod + eps) / (in_sq + tgt_sq + eps)

    return 1 - dice
