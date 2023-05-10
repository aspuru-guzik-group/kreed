#!/bin/sh
#SBATCH --mem=40G
#SBATCH -c 8
#SBATCH --partition=a40
#SBATCH --gres=gpu:1

#SBATCH --open-mode=append
#SBATCH --output=slurm/v_geom_1_gpu.out
#SBATCH --error=slurm/v_geom_1_gpu.err

python -m src.experimental.train --accelerator=gpu --devices=1 --num_workers=8 --dataset=geom --enable_wandb --wandb_run_id v_geom_1_gpu --check_samples_every_n_epoch 1 --batch_size 64 --checkpoint_dir=/checkpoint/${USER}/${SLURM_JOB_ID} --max_epochs=300

cp -r /checkpoint/${USER}/${SLURM_JOB_ID} ~/my_checkpoints
