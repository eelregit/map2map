#!/bin/bash

#SBATCH --job-name=srsgan
#SBATCH --output=%x-%j.out

#SBATCH --partition=rtx
##SBATCH --gres=gpu:4

#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00


hostname; pwd; date


#module load gcc python3
source $HOME/anaconda3/bin/activate


data_root_dir="/scratch1/06431/yueyingn/dmo-50MPC-train"

in_dir="low-resl"
tgt_dir="high-resl"

train_dirs="set[0-7]/output/PART_004"
#val_dirs="set4/output/PART_004"

in_files_1="disp.npy"
in_files_2="vel.npy"
tgt_files_1="disp.npy"
tgt_files_2="vel.npy"


srun m2m.py train \
    --train-in-patterns "$data_root_dir/$in_dir/$train_dirs/$in_files_1,$data_root_dir/$in_dir/$train_dirs/$in_files_2" \
    --train-tgt-patterns "$data_root_dir/$tgt_dir/$train_dirs/$tgt_files_1,$data_root_dir/$tgt_dir/$train_dirs/$tgt_files_2" \
    --in-norms cosmology.dis,cosmology.vel --tgt-norms cosmology.dis,cosmology.vel --augment --crop 88 --pad 20 --scale-factor 2 \
    --model VNet \
    --lr 0.0001 --batches 1 --loader-workers 0 \
    --epochs 1024 --seed $RANDOM


date
