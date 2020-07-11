import torch

class InstanceNoise:
    """Instance noise, with a linear decaying schedule
    """
    def __init__(self, init_std, batches):
        assert init_std >= 0, 'Noise std cannot be negative'
        self.init_std = init_std
        self._std = init_std
        self.batches = batches

    def std(self):
        self._std -= self.init_std / self.batches
        return max(self._std, 0)
