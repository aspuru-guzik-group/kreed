#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=sweep_train_edm
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --array=0-5%5
#SBATCH --output=logs/array-%A_%a.out
#SBATCH -c 8

source ~/.bashrc
conda activate egnn

IFS=$'\n' read -d '' -r -a lines < sweep_train_edm_qm9_jobs
cd ../..

echo "Starting task $SLURM_ARRAY_TASK_ID: ${lines[SLURM_ARRAY_TASK_ID]}"
echo ${lines[SLURM_ARRAY_TASK_ID]} --wandb_run_id ${SLURM_JOB_ID} --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}
eval ${lines[SLURM_ARRAY_TASK_ID]} --wandb_run_id ${SLURM_JOB_ID} --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}
