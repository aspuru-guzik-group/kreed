#!/bin/sh
#SBATCH --job-name=train_edm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --partition=t4v2
#SBATCH --qos=high

#SBATCH --open-mode=append
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err

srun python -m src.experimental.train --accelerator=gpu --devices=4 --num_nodes=1 --num_workers=16 --dataset=geom --enable_wandb --wandb_run_id ${SLURM_JOB_ID} --check_samples_every_n_epoch 1 --batch_size 32 --checkpoint_dir=/checkpoint/${USER}/${SLURM_JOB_ID} --max_epochs=100 --lr=8e-4 --strategy=ddp

cp -r /checkpoint/${USER}/${SLURM_JOB_ID} ~/my_checkpoints
