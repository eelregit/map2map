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
