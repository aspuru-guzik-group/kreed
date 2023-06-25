
qm9_chunks = 64
geom_chunks = 128
save_dir = "/scratch/ssd004/scratch/austin/baseline"
split = "test"

extra = " --enable_only_carbon_cond --pdropout_cond 0.1 0.1"

jobs_fname = "scripts/eval/baseline_jobs"

def make_baseline_jobs():
    with open("scripts/eval/array_jobs", "w") as array_jobs:
        dataset = "qm9"
        jobs = jobs_fname + "_qm9_p0"
        array_jobs.write(f"parallel < {jobs}\n")
        with open(jobs, "w") as f:
            for i in range(qm9_chunks):
                f.write(f"srun python -m src.experimental.evaluate_baseline --save_dir={save_dir} --split={split} --enable_save_samples_and_examples --num_chunks={qm9_chunks} --chunk_id={i} --dataset={dataset}\n")
        
        jobs = jobs_fname + "_qm9_p1"
        array_jobs.write(f"parallel < {jobs}\n")
        with open(jobs, "w") as f:
            for i in range(qm9_chunks):
                f.write(f"srun python -m src.experimental.evaluate_baseline --save_dir={save_dir} --split={split} --enable_save_samples_and_examples --num_chunks={qm9_chunks} --chunk_id={i} --dataset={dataset} {extra}\n")

        dataset = "geom"
        jobs = jobs_fname + "_geom_p0_first"
        array_jobs.write(f"parallel < {jobs}\n")
        with open(jobs, "w") as f:
            for i in range(geom_chunks // 2):
                f.write(f"srun python -m src.experimental.evaluate_baseline --save_dir={save_dir} --split={split} --enable_save_samples_and_examples --num_chunks={geom_chunks} --chunk_id={i} --dataset={dataset}\n")
        
        jobs = jobs_fname + "_geom_p0_second"
        array_jobs.write(f"parallel < {jobs}\n")
        with open(jobs, "w") as f:
            for i in range(geom_chunks // 2, geom_chunks):
                f.write(f"srun python -m src.experimental.evaluate_baseline --save_dir={save_dir} --split={split} --enable_save_samples_and_examples --num_chunks={geom_chunks} --chunk_id={i} --dataset={dataset}\n")
        
        jobs = jobs_fname + "_geom_p1_first"
        array_jobs.write(f"parallel < {jobs}\n")
        with open(jobs, "w") as f:
            for i in range(geom_chunks // 2):
                f.write(f"srun python -m src.experimental.evaluate_baseline --save_dir={save_dir} --split={split} --enable_save_samples_and_examples --num_chunks={geom_chunks} --chunk_id={i} --dataset={dataset} {extra}\n")
        
        jobs = jobs_fname + "_geom_p1_second"
        array_jobs.write(f"parallel < {jobs}\n")
        with open(jobs, "w") as f:
            for i in range(geom_chunks // 2, geom_chunks):
                f.write(f"srun python -m src.experimental.evaluate_baseline --save_dir={save_dir} --split={split} --enable_save_samples_and_examples --num_chunks={geom_chunks} --chunk_id={i} --dataset={dataset} {extra}\n")

if __name__ == "__main__":
    make_baseline_jobs()
