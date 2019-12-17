#!/bin/bash

#SBATCH --job-name=dis2dis-test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#SBATCH --partition=ccm

#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=1-00:00:00


hostname; pwd; date


module load gcc python3


export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
echo OMP_NUM_THREADS = $OMP_NUM_THREADS


data_root_dir="/mnt/ceph/users/yinli/Quijote"

in_dir="linear"
tgt_dir="nonlin"

test_dirs="0"  # FIXME

files="dis/512x???.npy"
in_files="$files"
tgt_files="$files"


m2m.py test \
    --test-in-patterns "$data_root_dir/$in_dir/$test_dirs/$in_files" \
    --test-tgt-patterns "$data_root_dir/$tgt_dir/$test_dirs/$tgt_files" \
    --norms cosmology.dis --model VNet --cache --crop 128 --pad 50 \
    --batches 1 --loader-workers 0 \
    --load-state best_model.pth


date
