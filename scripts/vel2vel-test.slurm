#!/bin/bash

#SBATCH --job-name=vel2vel-test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#SBATCH --partition=ccm

#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=1-00:00:00


hostname; pwd; date


module load gcc openmpi2
module load cuda/10.1.243_418.87.00 cudnn/v7.6.2-cuda-10.1

source $HOME/anaconda3/bin/activate torch


export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
echo OMP_NUM_THREADS = $OMP_NUM_THREADS


data_root_dir="/mnt/ceph/users/yinli/Quijote"

in_dir="linear"
tgt_dir="nonlin"

test_dirs="0"  # FIXME

files="vel/128x???.npy"
in_files="$files"
tgt_files="$files"


srun m2m.py test \
    --test-in-patterns "$data_root_dir/$in_dir/$test_dirs/$in_files" \
    --test-tgt-patterns "$data_root_dir/$tgt_dir/$test_dirs/$tgt_files" \
    --in-channels 3 --out-channels 3 --norms cosmology.vel \
    --batches 1 --loader-workers 0 --pad-or-crop 40 \
    --load-state best_model.pth


date
