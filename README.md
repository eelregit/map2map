# map2map
Neural network emulators to transform field/map data


* [Installation](#installation)
* [Usage](#usage)
    * [Data](#data)
        * [Data cropping](#data-cropping)
        * [Data padding](#data-padding)
        * [Data loading, sampling, and page caching](#data-loading-sampling-and-page-caching)
        * [Data normalization](#data-normalization)
    * [Model](#model)
    * [Training](#training)
        * [Files generated](#files-generated)
        * [Tracking](#tracking)
    * [Customization](#customization)


## Installation

Install in editable mode

```bash
pip install -e .
```


## Usage

The command is `m2m.py` in your `$PATH` after installation.
Take a look at the examples in `scripts/*.slurm`.
For all command line options look at `map2map/args.py` or do `m2m.py -h`.


### Data

Put each field in one npy file.
Structure your data to start with the channel axis and then the spatial
dimensions, e.g. `(2, 64, 64)` for a 2D vector field of size `64^2` and
`(1, 32, 32, 32)` for a 3D scalar field of size `32^3`.
Specify the data path with
[glob patterns](https://docs.python.org/3/library/glob.html).

During training, pairs of input and target fields are loaded.
Both input and target data can consist of multiple fields, which are
then concatenated along the channel axis.


#### Data cropping

If the size of a pair of input and target fields is too large to fit in
a GPU, we can crop part of them to form pairs of samples.
Each field can be cropped multiple times, along each dimension.
See `--crop`, `--crop-start`, `--crop-stop`, and `--crop-step`.
The total sample size is the number of input and target pairs multiplied
by the number of cropped samples per pair.


#### Data padding

Here we are talking about two types of padding.
We differentiate the padding during convolution from our explicit
data padding, and refer to the former as conv-padding.

Convolution preserves translational invariance, but conv-padding breaks
it, except for the periodic conv-padding, which is not feasible at
runtime for large 3D fields.
Therefore we recommend convolution without conv-padding.
By doing this, the output size will be smaller than the input size, and
thus smaller than the target size if it equals the input size, making
loss computation inefficient.

To solve this, we can pad the input before feeding it into the model.
The pad size should be adjusted so that the output size equals or
approximates the target size.
One should be able to calculate the proper pad size given the model.
Padding works for cropped samples, or samples with periodic boundary
condition.


#### Data loading, sampling, and page caching

The difference in speed between disks and GPUs makes training an
IO-bound job.
Stochastic optimization exacerbates the situation, especially for large
3D data *with multiple crops per field*.
In this case, we can use the `--div-data` option to divide field files
among GPUs, so that each node only need to load part of all data if
there are multiple nodes.
Data division is shuffled every epoch.
Crops within each field can be further randomized within a distance
relative to the field, controlled by `--div-shuffle-dist`.
Setting it to 0 turn off this randomization, and setting it to N limits
the shuffling within a distance of N files.
With both `--div-data` and `--div-shuffle-dist`, each GPU only need to
work on about N files at a time, with those files kept in the Linux page
cache.
This is especially useful when the amount of data exceeds the CPU memory
size.


#### Data normalization

Input and target (output) data can be normalized by functions defined in
`map2map2/data/norms/`.
Also see [Customization](#customization).


### Model

Find the models in `map2map/models/`.
Modify the existing models, or write new models somewhere and then
follow [Customization](#customization).

```python
class Net(nn.Module):
    def __init__(self, in_chan, out_chan, mid_chan=32, kernel_size=3,
                 negative_slope=0.2, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(in_chan, mid_chan, kernel_size)
        self.act = nn.LeakyReLU(negative_slope)
        self.conv2 = nn.Conv2d(mid_chan, out_chan, kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x
```

The model `__init__` requires two positional arguments, the number of
input and output channels.
Other hyperparameters can be specified as keyword arguments, including
the `scale_factor` useful for super-resolution tasks.
Note that the `**kwargs` is necessary for compatibility.


### Training


#### Files generated

* `*.out`: job stdout and stderr
* `state_{i}.pt`: training state after the i-th epoch including the
  model state
* `checkpoint.pt`: symlink to the latest state
* `runs/`: directories of tensorboard logs


#### Tracking

Install tensorboard and launch it by

```bash
tensorboard --logdir PATH --samples_per_plugin images=IMAGES --port PORT
```

* Use `.` as `PATH` in the training directory, or use the path to some parent
  directory for tensorboard to search recursively for multiple jobs.
* Show `IMAGES` images, or all of them by setting it to 0.
* Pick a free `PORT`. For remote jobs, do ssh port forwarding.


### Customization

Models, criteria, optimizers and data normalizations can be customized
without modifying map2map.
They can be implemented as callbacks in a user directory which is then
passed by `--callback-at`.
The default locations are searched first before the callback directory.
So be aware of name collisions.

The default locations are
* models: `map2map/models/`
* criteria: `torch.nn`
* optimizers: `torch.optim`
* normalizations: `map2map/data/norms/`

This approach is good for experimentation.
For example, one can play with a model `Bar` in `path/to/foo.py`, by
calling `m2m.py` with `--model foo.Bar --callback-at path/to`.
