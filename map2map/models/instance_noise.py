from math import log
import torch

class InstanceNoise:
    """Instance noise, with a heuristic decaying schedule
    """
    def __init__(self, init_std, batches):
        self.init_std = init_std
        self._std = init_std
        self.batches = batches

    def std(self, adv_loss):
        self._std -= self.init_std / self.batches
        self._std = self._std if self._std > 0 else 0
        return self._std
