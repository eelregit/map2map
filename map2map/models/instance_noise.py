from math import log
import torch

class InstanceNoise:
    """Instance noise, with a heuristic annealing schedule
    """
    def __init__(self, init_std):
        self.init_std = init_std
        self.adv_loss_cum = 0
        self.ln2 = log(2)
        self.batches = 1e5

    def std(self, adv_loss):
        self.adv_loss_cum += adv_loss
        anneal = 1 - self.adv_loss_cum / self.ln2 / self.batches
        anneal = anneal if anneal > 0 else 0
        std = anneal * self.init_std
        return std
