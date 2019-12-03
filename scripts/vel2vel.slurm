#!/bin/bash

#SBATCH --job-name=vel2vel
#SBATCH --dependency=singleton
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-32gb:4

#SBATCH --exclusive
#SBATCH --nodes=4
#SBATCH --mem=0
#SBATCH --time=7-00:00:00


hostname; pwd; date


module load gcc openmpi2
module load cuda/10.1.243_418.87.00 cudnn/v7.6.2-cuda-10.1

source $HOME/anaconda3/bin/activate torch


export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=8888


data_root_dir="/mnt/ceph/users/yinli/Quijote"

in_dir="linear"
tgt_dir="nonlin"

train_dirs="*[1-9]"
val_dirs="*[1-9]0"

files="vel/128x???.npy"
in_files="$files"
tgt_files="$files"


srun m2m.py train \
    --train-in-patterns "$data_root_dir/$in_dir/$train_dirs/$in_files" \
    --train-tgt-patterns "$data_root_dir/$tgt_dir/$train_dirs/$tgt_files" \
    --val-in-patterns "$data_root_dir/$in_dir/$val_dirs/$in_files" \
    --val-tgt-patterns "$data_root_dir/$tgt_dir/$val_dirs/$tgt_files" \
    --in-channels 3 --out-channels 3 --norms cosmology.vel --augment \
    --epochs 1024 --batches 3 --loader-workers 3 --lr 0.001
#    --load-state checkpoint.pth


date
