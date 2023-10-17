#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=eval
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --partition=a40,rtx6000
#SBATCH --qos=normal
#SBATCH --array=0-4
#SBATCH --output=logs/qm9_rot_only/%A_%a.out
#SBATCH -c 8

mkdir -p logs/qm9_rot_only

source ~/.bashrc
conda activate egnn

K=100
N_CHUNKS=50
NAME=qm9
SPLIT=test
BATCH_SIZE=10
SAVE_DIR=samples

srun python -m src.experimental.evaluate --num_workers=8 --chunk_id=${SLURM_ARRAY_TASK_ID} --num_chunks=$N_CHUNKS --checkpoint_path=final_checkpoints/$NAME.ckpt --save_dir=$SAVE_DIR/rot_only/$NAME --enable_save_samples_and_examples --pdropout_cond 1.0 1.0 --split=$SPLIT --batch_size=$BATCH_SIZE --sample_n_candidates_per_example=$K
