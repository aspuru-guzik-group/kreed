#!/bin/bash

#SBATCH --job-name=train_edm
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=logs/slurm-%j.out

export NCCL_SOCKET_IFNAME=bond0

# Conda
source ~/.bashrc
conda activate egnn

cd ../..

srun --nodes=2 --tasks-per-node=4  python -m src.experimental.train \
  --accelerator=gpu --nodes=2  --devices=4 \
  --batch_size=16 --num_workers=8 --dataset=geom \
  --max_epochs=100 --check_samples_every_n_epochs=5 --samples_assess_n_batches=10 --samples_visualize_n_mols=0 \
  --hidden_features=512 --inner_features=512 \
  --enable_wandb  --wandb_project=train_geom_large --wandb_run_id ${SLURM_JOB_ID} --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}
