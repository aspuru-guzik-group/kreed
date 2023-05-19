#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=eval_geom
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --partition=rtx6000,a40
#SBATCH --qos=normal
#SBATCH --array=550-1000%64
#SBATCH --output=slurm/eval_array/p10_%A_%a.out
#SBATCH -c 8

mkdir -p slurm/eval_array

source ~/.bashrc
conda activate egnn

ROOT=~/geom_chunks
CKPT=final_checkpoints/geom.ckpt
NSAMPLES=10
K=5
THRESHOLD=0.05
p=0.10
DIRECTORY=$ROOT/p${p}

srun python scripts/eval/generate_eval_chunk_geom_samples.py --directory=$DIRECTORY --checkpoint_path=$CKPT --p_drop=$p --samples_per_example=$NSAMPLES --split=test --k=$K --threshold=$THRESHOLD --chunk=${SLURM_ARRAY_TASK_ID}
