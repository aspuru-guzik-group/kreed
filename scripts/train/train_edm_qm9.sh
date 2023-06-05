#!/bin/bash

#SBATCH --job-name=train_edm
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=logs/slurm-%j.out

# Conda
source ~/.bashrc
conda activate egnn

cd ../..

srun python -m src.experimental.train --accelerator=gpu --devices=1 --num_workers=8 --dataset=qm9 --enable_wandb --wandb_project=sweep_train_edm_qm9_debug --max_epochs=10 --check_samples_every_n_epochs=1 --samples_assess_n_batches=1 --samples_visualize_n_mols=0 --parameterization=eps --egnn_equivariance=e3   --wandb_run_name=eps-e3-ortho
