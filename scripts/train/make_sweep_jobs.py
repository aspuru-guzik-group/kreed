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

source ~/.bashrc
conda activate egnn

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
    "--dataset=qm9 --enable_wandb --wandb_project=sweep_architecture_train_qm9_uniform --pdropout_cond 0.0 1.0 "
    "--max_epochs=4000 "
    "--check_samples_every_n_epochs=200 --samples_assess_n_batches=10 --samples_visualize_n_mols=0"
)

SWEEP_GRID = [
    # [("eps", "--parameterization=eps"), ("x", "--parameterization=x")],
    # [("e3", "--egnn_equivariance=e3"), ("rfl", "--egnn_equivariance=ref")],
    # [("rlx", ""), ("", "--disable_egnn_relaxed")],
    # [("ortho", ""), ("trnsl", "--disable_project_sample_orthogonally")],
    # ["--norm_type=layer", "--norm_type=none --disable_norm_adaptively"],
    [
        ("edm-e3", "--architecture=edm --egnn_equivariance=e3 --disable_egnn_relaxed"),
        ("edm-ref", "--architecture=edm --disable_egnn_relaxed"),
        ("tf-ref", "--disable_norm_adaptively --disable_egnn_relaxed"),
        ("tf-ref-rlx", "--disable_norm_adaptively"),
        ("tf-ref-rlx-ada", ""),
    ]
]


def make_sweep_jobs():
    jobs_fname = "sweep_train_edm_qm9_jobs"
    jobs_size = 0

    with open(jobs_fname, "w+") as f:
        for names_and_flags in itertools.product(*SWEEP_GRID):
            names, flags = zip(*names_and_flags)
            run_name = "-".join(filter(None, names))
            parts = [TRAIN_COMMAND_TEMPLATE] + list(flags) + [f"--wandb_run_name={run_name}"]
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
