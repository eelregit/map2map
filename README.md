# map2map
Neural network emulators to transform field/map data


* [Installation](#installation)
* [Usage](#usage)
    * [Data](#data)
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
dimensions.
For example a 2D vector field of size `64^2` should have shape `(2, 64,
64)`.
Specify the data path with
[glob patterns](https://docs.python.org/3/library/glob.html).

During training, pairs of input and target fields are loaded.
Both input and target data can consist of multiple fields, which are
then concatenated along the channel axis.
If the size of a pair of input and target fields is too large to fit in
a GPU, we can crop part of them to form pairs of samples (see `--crop`).
Each field can be cropped multiple times, along each dimension,
controlled by the spacing between two adjacent crops (see `--step`).
The total sample size is the number of input and target pairs multiplied
by the number of cropped samples per pair.


#### Data normalization

Input and target (output) data can be normalized by functions defined in
`map2map2/data/norms/`.
Also see [Customization](#customization).


### Model

Find the models in `map2map/models/`.
Modify the existing models, or write new models somewhere and then
follow [Customization](#customization).


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

This approach is good for experimentation.
For example, one can play with a model `Bar` in `path/to/foo.py`, by
calling `m2m.py` with `--model foo.Bar --callback-at path/to`.
