#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=eval_geom
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --partition=rtx6000,a40
#SBATCH --qos=normal
#SBATCH --output=slurm/eval_array/p10_%A_%a.out
#SBATCH -c 8

mkdir -p slurm/eval_array

source ~/.bashrc
conda activate egnn

ROOT=~/qm9_chunks
CKPT=final_checkpoints/qm9_p10_e3.ckpt
NSAMPLES=10
K=5
THRESHOLD=0.05
p=0.10
DIRECTORY=$ROOT/p${p}

srun python scripts/eval/generate_eval_all_qm9_samples.py \
  --directory=$DIRECTORY --checkpoint_path=$CKPT \
  --p_drop=$p --samples_per_example=$NSAMPLES \
  --split=test --k=$K --threshold=$THRESHOLD
