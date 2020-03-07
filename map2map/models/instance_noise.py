from math import log
import torch

class InstanceNoise:
    """Instance noise, with a heuristic annealing schedule
    """
    def __init__(self, init_std):
        self.init_std = init_std
        self.anneal = 1
        self.ln2 = log(2)
        self.batches = 1e5

    def std(self, adv_loss):
        self.anneal -= adv_loss / self.ln2 / self.batches
        std = self.anneal * self.init_std
        std = std if std > 0 else 0
        return std
