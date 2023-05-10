#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=eval_geom
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --partition=rtx6000,a40
#SBATCH --qos=normal
#SBATCH --array=10-250%16
#SBATCH --output=slurm/array-%A_%a.out
#SBATCH --error=slurm/array-%A_%a.err
#SBATCH -c 8

source ~/.bashrc
conda activate egnn

ROOT=/h/austin/geom_chunks
CKPT=final_checkpoints/geom.ckpt
NSAMPLES=10
K=5
THRESHOLD=0.05
p=0.0
DIRECTORY=$ROOT/p${p}

srun python scripts/eval/generate_eval_chunk_geom_samples.py --directory=$DIRECTORY --checkpoint_path=$CKPT --p_drop=$p --samples_per_example=$NSAMPLES --split=test --k=$K --threshold=$THRESHOLD --chunk=${SLURM_ARRAY_TASK_ID}
