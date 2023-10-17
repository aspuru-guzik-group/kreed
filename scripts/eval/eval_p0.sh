#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=eval
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --partition=rtx6000,a40
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --output=logs/%A_%a.out
#SBATCH -c 8

mkdir -p logs

source ~/.bashrc
conda activate egnn

N_CHUNKS=8
NAME=qm9 # or geom
SPLIT=test
BATCH_SIZE=100
SAVE_DIR=samples

srun python -m src.experimental.evaluate --num_workers=8 --chunk_id=${SLURM_ARRAY_TASK_ID} --num_chunks=$N_CHUNKS --checkpoint_path=final_checkpoints/$NAME.ckpt --save_dir=$SAVE_DIR/p0/$NAME --enable_save_samples_and_examples --split=$SPLIT --batch_size=$BATCH_SIZE
