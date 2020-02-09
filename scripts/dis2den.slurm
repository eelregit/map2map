#!/bin/bash

#SBATCH --job-name=dis2den
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

in_files="dis.npy"
tgt_files="den.npy"


srun m2m.py train \
    --train-in-patterns "$data_root_dir/$in_dir/$train_dirs/$in_files" \
    --train-tgt-patterns "$data_root_dir/$tgt_dir/$train_dirs/$tgt_files" \
    --val-in-patterns "$data_root_dir/$in_dir/$val_dirs/$in_files" \
    --val-tgt-patterns "$data_root_dir/$tgt_dir/$val_dirs/$tgt_files" \
    --in-norms cosmology.dis --tgt-norms torch.log1p --augment --crop 128 --pad 20 \
    --model UNet \
    --lr 0.0001 --batches 1 --loader-workers 0 \
    --epochs 1024 --seed $RANDOM \
    --cache --div-data
#    --load-state checkpoint.pth \


date
