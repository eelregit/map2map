#!/bin/bash

#SBATCH --job-name=R2D2
#SBATCH --output=%x-%j.out

#SBATCH --partition=cpu_partition

#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --time=1-00:00:00


hostname; pwd; date


# set computing environment, e.g. with module or anaconda
#module load python
#source $HOME/anaconda3/bin/activate pytorch_env


export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE  # use MKL-DNN


m2m.py test \
    --test-in-patterns "test/R0-*.npy,test/R1-*.npy" \
    --test-tgt-patterns "test/D0-*.npy,test/D1-*.npy" \
    --in-norms RnD.R0,RnD.R1 --tgt-norms RnD.D0,RnD.D1 \
    --model model.Net --callback-at . \
    --batches 1 \
    --load-state checkpoint.pt


date