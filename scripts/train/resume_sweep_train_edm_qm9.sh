#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=sweep_train_edm
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --array=0-7%7
#SBATCH --output=logs/array-%A_%a.out
#SBATCH -c 8

source ~/.bashrc
conda activate egnn

IFS=$'\n' read -d '' -r -a lines < resume_sweep_train_edm_qm9_jobs
cd ../..

echo "Starting task $SLURM_ARRAY_TASK_ID: ${lines[SLURM_ARRAY_TASK_ID]}"
echo ${lines[SLURM_ARRAY_TASK_ID]}
eval ${lines[SLURM_ARRAY_TASK_ID]}
