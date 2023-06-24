
qm9_chunks = 100
geom_chunks = 150
save_dir = "/scratch/ssd004/scratch/austin/baseline"
split = "test"

extra = " --enable_only_carbon_cond --pdropout_cond 0.1 0.1"

jobs_fname = "scripts/eval/baseline_jobs.txt"

def make_baseline_jobs():

    with open(jobs_fname, "w") as f:
        dataset = 'qm9'
        for i in range(qm9_chunks):
            f.write(f"srun python -m src.experimental.evaluate_baseline --save_dir={save_dir} --split={split} --enable_save_samples_and_examples --num_chunks={qm9_chunks} --chunk_id={i} --dataset={dataset}\n")
            f.write(f"srun python -m src.experimental.evaluate_baseline --save_dir={save_dir} --split={split} --enable_save_samples_and_examples --num_chunks={qm9_chunks} --chunk_id={i} --dataset={dataset} {extra}\n")
        
        dataset = 'geom'
        for i in range(geom_chunks):
            f.write(f"srun python -m src.experimental.evaluate_baseline --save_dir={save_dir} --split={split} --enable_save_samples_and_examples --num_chunks={geom_chunks} --chunk_id={i} --dataset={dataset}\n")
            f.write(f"srun python -m src.experimental.evaluate_baseline --save_dir={save_dir} --split={split} --enable_save_samples_and_examples --num_chunks={geom_chunks} --chunk_id={i} --dataset={dataset} {extra}\n")

if __name__ == "__main__":
    make_baseline_jobs()
