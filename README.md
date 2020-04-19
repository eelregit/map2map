# map2map
Neural network emulators to transform field/map data


## Installation

Install in editable mode

```bash
pip install -e .
```


## Usage

Take a look at the examples in `scripts/*.slurm`, and the command line options
in `map2map/args.py` or by `m2m.py -h`.


### data

Structure your data to start with the channel axis and then the spatial
dimensions.
Put each sample in one file.
Specify the data path with glob patterns.


#### data normalization

Input and target (output) data can be normalized by functions defined in
`map2map2/data/norms/`.


### model

Find the models in `map2map/models/`.
Customize the existing models, or add new models there and edit the `__init__.py`.


### Training


#### Files generated

* `*.out`: job stdout and stderr
* `state_*.pth`: training state including the model parameters
* `checkpoint.pth`: symlink to the latest state
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
