#!/bin/bash

#SBATCH --job-name=dis2dis-test
#SBATCH --output=%x-%j.out

#SBATCH --partition=ccm

#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00


hostname; pwd; date


#module load gcc python3
source $HOME/anaconda/bin/activate torch


export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE


data_root_dir="/mnt/ceph/users/yinli/Quijote"

in_dir="linear"
tgt_dir="nonlin"

test_dirs="*99"

files="dis.npy"
in_files="$files"
tgt_files="$files"


m2m.py test \
    --test-in-patterns "$data_root_dir/$in_dir/$test_dirs/$in_files" \
    --test-tgt-patterns "$data_root_dir/$tgt_dir/$test_dirs/$tgt_files" \
    --norms cosmology.dis --crop 256 --pad 20 \
    --model VNet \
    --load-state best_model.pth \
    --batches 1 --loader-workers 0 \
    --cache


date
