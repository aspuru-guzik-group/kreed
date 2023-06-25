#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=eval_baseline
#SBATCH --mem=0
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --array=1-6
#SBATCH --output=logs/eval_baseline/%A_%a.out
#SBATCH -c 64

source ~/.bashrc
conda activate egnn

eval "$(sed -n ${SLURM_ARRAY_TASK_ID}p scripts/eval/array_jobs)"
