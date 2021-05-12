import torch


def exp(x, undo=False, **kwargs):
    if not undo:
        torch.exp(x, out=x)
    else:
        torch.log(x, out=x)

def log(x, eps=1e-8, undo=False, **kwargs):
    if not undo:
        torch.log(x + eps, out=x)
    else:
        torch.exp(x, out=x)

def expm1(x, undo=False, **kwargs):
    if not undo:
        torch.expm1(x, out=x)
    else:
        torch.log1p(x, out=x)

def log1p(x, eps=1e-7, undo=False, **kwargs):
    if not undo:
        torch.log1p(x + eps, out=x)
    else:
        torch.expm1(x, out=x)
