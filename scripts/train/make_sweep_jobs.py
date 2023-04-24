import itertools

SLURM_TEMPLATE = (
"""#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=sweep_train_edm
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --array=0-{num_jobs}%{num_parallel_jobs}
#SBATCH --output=logs/array-%A_%a.out
#SBATCH -c 8

export LD_LIBRARY_PATH=/pkgs/cuda-11.8/lib64:/pkgs/cudnn-11.7-v8.5.0.96/lib64:$LD_LIBRARY_PATH
export CUDA_PATH=/pkgs/cuda-11.8/
export CUDNN_PATH=/pkgs/cudnn-11.7-v8.5.0.96/

source ~/.bashrc
conda activate edm

IFS=$'\\n' read -d '' -r -a lines < {job_fname}
cd ../..

echo "Starting task $SLURM_ARRAY_TASK_ID: ${{lines[SLURM_ARRAY_TASK_ID]}}"
echo ${{lines[SLURM_ARRAY_TASK_ID]}} --wandb_run_id ${{SLURM_JOB_ID}} --checkpoint_dir /checkpoint/${{USER}}/${{SLURM_JOB_ID}}
eval ${{lines[SLURM_ARRAY_TASK_ID]}} --wandb_run_id ${{SLURM_JOB_ID}} --checkpoint_dir /checkpoint/${{USER}}/${{SLURM_JOB_ID}}
"""
)

TRAIN_COMMAND_TEMPLATE = (
    "srun python -m src.experimental.train "
    "--accelerator=gpu --devices=1 --num_workers=8 "
    "--dataset=qm9 --enable_wandb "
    "--check_samples_every_n_epochs=50 --samples_assess_n_batches=10 --samples_visualize_n_mols=0"
)

SWEEP_GRID = [
    ["--parameterization=eps", "--parameterization=x"],
    ["--norm_type=graph", "--norm_type=graph --disable_norm_adaptively", "--norm_type=none --disable_norm_adaptively"],
    ["--egnn_equivariance=e3", "--egnn_equivariance=ref"],
    ["", "--disable_egnn_relaxed"],
]


def make_sweep_jobs():
    jobs_fname = "sweep_train_edm_qm9_jobs"
    jobs_size = 0

    with open(jobs_fname, "w+") as f:
        for flags in itertools.product(*SWEEP_GRID):
            parts = [TRAIN_COMMAND_TEMPLATE] + list(flags)
            f.write(" ".join(parts) + "\n")
            jobs_size += 1

    with open(f"sweep_train_edm_qm9.sh", "w+") as f:
        f.write(SLURM_TEMPLATE.format(
            job_fname=f"sweep_train_edm_qm9_jobs",
            num_jobs=jobs_size,
            num_parallel_jobs=min(40, jobs_size),
        ))


if __name__ == "__main__":
    make_sweep_jobs()
