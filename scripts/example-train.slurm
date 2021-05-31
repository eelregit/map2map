#!/bin/bash

#SBATCH --job-name=R2D2
#SBATCH --output=%x-%j.out
#SBATCH --partition=gpu_partition
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=1-00:00:00


echo "This is a minimal example. See --help or args.py for more," \
     "e.g. on augmentation, cropping, padding, and data division."
echo "Training on 2 nodes with 8 GPUs."
echo "input data: {train,val,test}/R{0,1}-*.npy"
echo "target data: {train,val,test}/D{0,1}-*.npy"
echo "normalization functions: {R,D}{0,1} in ./RnD.py," \
     "see map2map/data/norms/*.py for examples"
echo "model: Net in ./model.py, see map2map/models/*.py for examples"
echo "Training with placeholder learning rate 1e-4 and batch size 1."


hostname; pwd; date


# set computing environment, e.g. with module or anaconda

#module load python
#module list

#source $HOME/anaconda3/bin/activate pytorch_env
#conda info


srun m2m.py train \
    --train-in-patterns "train/R0-*.npy,train/R1-*.npy" \
    --train-tgt-patterns "train/D0-*.npy,train/D1-*.npy" \
    --val-in-patterns "val/R0-*.npy,val/R1-*.npy" \
    --val-tgt-patterns "val/D0-*.npy,val/D1-*.npy" \
    --in-norms RnD.R0,RnD.R1 --tgt-norms RnD.D0,RnD.D1 \
    --model model.Net --callback-at . \
    --lr 1e-4 --batch-size 1 \
    --epochs 1024 --seed $RANDOM


date
