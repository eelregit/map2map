from itertools import chain
import torch
from torch.utils.data import Sampler


class GroupedRandomSampler(Sampler):
    """Sample randomly within each group of samples and sequentially from group
    to group.

    This behaves like a simple random sampler by default
    """

    def __init__(self, data_source, group_size=None):
        self.data_source = data_source
        self.sample_size = len(data_source)

        if group_size is None:
            group_size = self.sample_size
        self.group_size = group_size

    def __iter__(self):
        starts = range(0, self.sample_size, self.group_size)
        sizes = [self.group_size] * (len(starts) - 1)
        sizes.append(self.sample_size - starts[-1])

        return iter(
            chain(
                *[
                    (start + torch.randperm(size)).tolist()
                    for start, size in zip(starts, sizes)
                ]
            )
        )

    def __len__(self):
        return self.sample_size
