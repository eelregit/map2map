#!/bin/bash

echo "This is the 10th test run of srsgan on 2D images. See --help or args.py for more," \
     "e.g. on augmentation, cropping, padding, and data division."
echo "Training on 1 nodes with 1 GPU(s)."
echo "input data: LR-{train,val}/*{gas,DM}.npy"
echo "target data: HR-{train,val}/*{gas,DM}.npy"
echo "normalization functions: {LR,HR}{G,D} in ./LNs.py," \
     "see map2map/data/norms/*.py for examples"
echo "model: Net in ./2Dsrsgan.py, see map2map/models/*.py for examples"
echo "Training with placeholder learning rate 1e-5 and batch size 2."


hostname; pwd; date


# set computing environment, e.g. with module or anaconda
#module load python
#source $HOME/anaconda3/bin/activate pytorch_env

export SLURM_JOB_NUM_NODES=1
export SLURM_NODE_ID=0
export SLURM_NODEID=0

m2m.py train \
    --train-in-patterns "/home/plachanc/data/finalized_data/LR-train/*gas.npy","/home/plachanc/data/finalized_data/LR-train/*DM.npy" \
    --train-tgt-patterns "/home/plachanc/data/finalized_data/HR-train/*gas.npy","/home/plachanc/data/finalized_data/HR-train/*DM.npy"  \
    --val-in-patterns "/home/plachanc/data/finalized_data/LR-val/*gas.npy","/home/plachanc/data/finalized_data/LR-val/*DM.npy" \
    --val-tgt-patterns "/home/plachanc/data/finalized_data/HR-val/*gas.npy","/home/plachanc/data/finalized_data/HR-val/*DM.npy" \
    --in-norms LNs.LRG,LNs.LRD --tgt-norms LNs.HRG,LNs.HRD \
    --scale-factor 8 --augment \
    --model 2Dsrsgan.G2 --adv-model 2Dsrsgan.D2 \
    --criterion MSELoss \
    --cgan --adv-start 5 \
    --adv-wgan-gp-interval 1 \
    --lr 1e-4 --optimizer-args '{"betas": [0, 0.99]}' \
    --adv-lr 1e-5 \
    --batch-size 1 --loader-workers 32 \
    --epochs 1000 \
    --adv-iter-ratio 5 \
    --callback-at . \


date
