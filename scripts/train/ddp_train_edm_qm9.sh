#!/bin/bash

#SBATCH --job-name=ddp_train_edm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=logs/slurm-%j.out

# Environment variables
export LD_LIBRARY_PATH=/pkgs/cuda-11.8/lib64:/pkgs/cudnn-11.7-v8.5.0.96/lib64:$LD_LIBRARY_PATH
export CUDA_PATH=/pkgs/cuda-11.8/
export CUDNN_PATH=/pkgs/cudnn-11.7-v8.5.0.96/

# Conda
source ~/.bashrc
conda activate edm

cd ../..

srun python -m src.experimental.train \
  --strategy=ddp --accelerator=gpu --num_nodes=1 --devices=4 --num_workers=8  \
  --dataset=qm9  --lr=1e-3 \
  --enable_wandb --wandb_run_id ${SLURM_JOB_ID} --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID} \
  --check_samples_every_n_epochs=50