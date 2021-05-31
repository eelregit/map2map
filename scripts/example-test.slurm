#!/bin/bash

#SBATCH --job-name=R2D2
#SBATCH --output=%x-%j.out
#SBATCH --partition=cpu_partition
#SBATCH --nodes=1
#SBATCH --exclusive
##SBATCH --partition=gpu_partition
##SBATCH --gres=gpu:1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=8
#SBATCH --time=0-01:00:00


hostname; pwd; date


# set computing environment, e.g. with module or anaconda

#module load python
#module list

#source $HOME/anaconda3/bin/activate pytorch_env
#conda info


m2m.py test \
    --test-in-patterns "test/R0-*.npy,test/R1-*.npy" \
    --test-tgt-patterns "test/D0-*.npy,test/D1-*.npy" \
    --in-norms RnD.R0,RnD.R1 --tgt-norms RnD.D0,RnD.D1 \
    --model model.Net --callback-at . \
    --batch-size 1 \
    --load-state checkpoint.pt


date
