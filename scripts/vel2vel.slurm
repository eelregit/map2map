#!/bin/bash

#SBATCH --job-name=vel2vel
#SBATCH --output=%x-%j.out

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-32gb:4

#SBATCH --exclusive
#SBATCH --nodes=4
#SBATCH --time=7-00:00:00


hostname; pwd; date


module load gcc python3
#source $HOME/anaconda3/bin/activate torch


export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=60606


data_root_dir="/mnt/ceph/users/yinli/Quijote"

in_dir="linear"
tgt_dir="nonlin"

train_dirs="*[0-8]"
val_dirs="*[0-8]9"

files="vel.npy"
in_files="$files"
tgt_files="$files"


srun m2m.py train \
    --train-in-patterns "$data_root_dir/$in_dir/$train_dirs/$in_files" \
    --train-tgt-patterns "$data_root_dir/$tgt_dir/$train_dirs/$tgt_files" \
    --val-in-patterns "$data_root_dir/$in_dir/$val_dirs/$in_files" \
    --val-tgt-patterns "$data_root_dir/$tgt_dir/$val_dirs/$tgt_files" \
    --in-norms cosmology.vel --tgt-norms cosmology.vel --augment --crop 128 --pad 20 \
    --model VNet --adv-model UNet --cgan \
    --lr 0.0001 --adv-lr 0.0004 --batches 1 --loader-workers 0 \
    --epochs 128 --seed $RANDOM \
    --cache --div-data
#    --load-state checkpoint.pth \


date
