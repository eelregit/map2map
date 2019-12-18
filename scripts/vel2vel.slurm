#!/bin/bash

#SBATCH --job-name=vel2vel
#SBATCH --dependency=singleton
#SBATCH --output=%x-%j.out

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-32gb:4

#SBATCH --exclusive
#SBATCH --nodes=4
#SBATCH --time=7-00:00:00


hostname; pwd; date


module load gcc python3


export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=8888


data_root_dir="/mnt/ceph/users/yinli/Quijote"

in_dir="linear"
tgt_dir="nonlin"

train_dirs="*[0-8]"
val_dirs="*[0-8]9"

files="vel/512x000.npy"
in_files="$files"
tgt_files="$files"


srun m2m.py train \
    --train-in-patterns "$data_root_dir/$in_dir/$train_dirs/$in_files" \
    --train-tgt-patterns "$data_root_dir/$tgt_dir/$train_dirs/$tgt_files" \
    --val-in-patterns "$data_root_dir/$in_dir/$val_dirs/$in_files" \
    --val-tgt-patterns "$data_root_dir/$tgt_dir/$val_dirs/$tgt_files" \
    --norms cosmology.vel --augment --crop 100 --pad 42 \
    --model VNet \
#    --load-state checkpoint.pth \
    --epochs 128 --lr 0.001 --batches 1 --loader-workers 0


date
