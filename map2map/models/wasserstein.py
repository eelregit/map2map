import torch
import torch.nn as nn


class WDistLoss(nn.Module):
    """Wasserstein distance

    target should have values of 0 (False) or 1 (True)
    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return wasserstein_distance_loss(input, target)


def wasserstein_distance_loss(input, target):
    sign = 2 * target - 1

    return - (sign * input).mean()


def wgan_grad_penalty(critic, x, y, lam=10):
    """Calculate the gradient penalty for WGAN
    """
    device = x.device
    batch_size = x.shape[0]
    alpha = torch.rand(batch_size, device=device)
    alpha = alpha.reshape(batch_size, *(1,) * (x.dim() - 1))

    xy = alpha * x.detach() + (1 - alpha) * y.detach()

    score = critic(xy.requires_grad_(True))
    # average over spatial dimensions if present
    score = score.flatten(start_dim=1).mean(dim=1)
    # sum over batches because graphs are mostly independent (w/o batchnorm)
    score = score.sum()

    grad, = torch.autograd.grad(
        score,
        xy,
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )

    grad = grad.flatten(start_dim=1)
    penalty = (
        lam * ((grad.norm(p=2, dim=1) - 1) ** 2).mean()
        + 0 * score  # hack to trigger DDP allreduce hooks
    )

    return penalty
