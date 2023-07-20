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

cp -r /ssd005/projects/acshare/10502448 /checkpoint/${USER}

srun --nodes=3 --tasks-per-node=4 python -m src.experimental.train --num_nodes=3 --accelerator=gpu --devices=4 --num_workers=10 --dataset=geom --enable_wandb --wandb_project=transformer_geom --max_epochs=400 --check_samples_every_n_epochs=10 --samples_visualize_n_mols=0 --batch_size=60 --wandb_run_name=v_transformer_rtx_huge_lr3_uniform --lr=3e-4 --pdropout_cond 0.0 1.0 --wandb_run_id 10502448 --checkpoint_dir /checkpoint/${USER}/10502448

cp -r /checkpoint/${USER}/10502448 /ssd005/projects/acshare/done
