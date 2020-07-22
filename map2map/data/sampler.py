import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class DistFieldSampler(Sampler):
    """Distributed sampler for field data, useful for multiple crops

    Stochastic training on fields with multiple crops puts burden on the IO.
    A node may load files of the whole field but only need a small part of it.
    Numpy memmap can load part of the field, but can also be very slow (even
    slower than reading the whole thing)

    `div_data` enables data file division among GPUs when `shuffle=True`.
    For field with multiple crops, it helps IO by benefiting from the page
    cache, but limits stochasticity.
    Increase `div_shuffle_dist` can mitigate this by shuffling the order of
    samples within the specified distance.

    When `div_data=False` this sampler behaves similar to `DistributedSampler`,
    except for the chunky (rather than strided) subsample slicing.
    Like `DistributedSampler`, `set_epoch()` should be called at the beginning
    of each epoch during training.
    """
    def __init__(self, dataset, shuffle,
                 div_data=False, div_shuffle_dist=0):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dataset = dataset
        self.nsample = len(dataset)
        self.nfile = dataset.nfile
        self.ncrop = dataset.ncrop

        self.shuffle = shuffle

        self.div_data = div_data
        self.div_shuffle_dist = div_shuffle_dist

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)

            if self.div_data:
                # shuffle files
                ind = torch.randperm(self.nfile, generator=g)
                ind = ind[:, None] * self.ncrop + torch.arange(self.ncrop)
                ind = ind.flatten()

                # displace crops with respect to files
                dis = torch.rand((self.nfile, self.ncrop),
                                 generator=g) * self.div_shuffle_dist
                loc = torch.arange(self.nfile)
                loc = loc[:, None] + dis
                loc = loc.flatten() % self.nfile  # periodic in files
                dis_ind = loc.argsort()

                # shuffle crops
                ind = ind[dis_ind].tolist()
            else:
                ind = torch.randperm(self.nsample, generator=g).tolist()
        else:
            ind = list(range(self.nsample))

        start = self.rank * len(self)
        stop = start + len(self)
        ind = ind[start:stop]

        return iter(ind)

    def __len__(self):
        return self.nsample // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch
