#!/bin/bash

#SBATCH --job-name=train_v
#SBATCH --nodes=3
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=logs/slurm-%j.out

export NCCL_SOCKET_IFNAME=bond0
#export NCCL_DEBUG=INFO

# Conda
source ~/.bashrc
conda activate egnn

srun --nodes=3 --tasks-per-node=4 python -m src.experimental.train --num_nodes=3 --accelerator=gpu --devices=4 --num_workers=10 --dataset=geom --enable_wandb --wandb_project=transformer_geom --max_epochs=100 --check_samples_every_n_epochs=5 --batch_size=64 --wandb_run_name=v_transformer_rtx_huge --lr=2.5e-4 --checkpoint_dir=/checkpoint/${USER}/${SLURM_JOB_ID} --strategy=ddp --wandb_run_id=${SLURM_JOB_ID}

cp /checkpoint/${USER}/${SLURM_JOB_ID} ${SCRATCH} -r

