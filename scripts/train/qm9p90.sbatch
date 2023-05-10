#!/bin/bash

#SBATCH --job-name=train_edm
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --output=slurm/slurm-%j.out

srun python -m src.experimental.train --accelerator=gpu --devices=1 --num_workers=8 --dataset=qm9 --enable_wandb --wandb_run_id ${SLURM_JOB_ID} --check_samples_every_n_epoch 200 --max_epochs=3000 --pdropout_cond=0.9 --checkpoint_dir=/checkpoint/${USER}/${SLURM_JOB_ID}

cp -r /checkpoint/${USER}/${SLURM_JOB_ID} ~/my_checkpoints


