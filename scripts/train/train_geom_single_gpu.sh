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

srun python -m src.experimental.train \
  --accelerator=gpu --devices=1 \
  --batch_size=8 --num_workers=8 --dataset=geom \
  --max_epochs=100 --check_samples_every_n_epochs=5 --samples_assess_n_batches=1 --samples_visualize_n_mols=0 \
  --hidden_features=512 --inner_features=512 \
  --enable_wandb  --wandb_project=train_geom_large --wandb_run_id ${SLURM_JOB_ID} --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}
