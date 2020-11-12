#!/bin/bash



hostname; pwd; date


# set computing environment, e.g. with module or anaconda
#module load python
#source $HOME/anaconda3/bin/activate pytorch_env

export SLURM_JOB_NUM_NODES=1
export SLURM_NODE_ID=0
export SLURM_NODEID=0


m2m.py test \
    --test-in-patterns "/home/plachanc/data/finalized_data/LR-val/*gas.npy","/home/plachanc/data/finalized_data/LR-val/*DM.npy" \
    --test-tgt-patterns "/home/plachanc/data/finalized_data/HR-val/*gas.npy","/home/plachanc/data/finalized_data/HR-val/*DM.npy"  \
    --in-norms LNs.LRG,LNs.LRD --tgt-norms LNs.HRG,LNs.HRD --callback-at . \
    --scale-factor 8 \
    --model 2Dsrsgan.G2 --callback-at . \
    --batch-size 1 \
    --load-state checkpoint.pt


date
