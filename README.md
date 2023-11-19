# kreed

KREED: **K**raitchman **RE**flection-**E**quivariant **D**iffusion

Code for Reflection-Equivariant Diffusion for 3D Structure Determination from Isotopologue Rotational Spectra in Natural Abundance

For a ready-to-use demonstration of the trained model, check out the [Colab notebook](https://colab.research.google.com/drive/17OWnUfGW8zqTdurAPahCGok7sxWP3Rza?usp=sharing)

## Training the model
Follow instructions in SETUP.md to setup the QM9 and GEOM datasets.
Preprocessed datasets and generated samples can be found [here](https://drive.google.com/drive/folders/1eRA5-Z42gSkw5IJobADGPJT_2lPMogOf?usp=sharing)


Setting up conda environment:
```
conda env create -f environment.yml
```

Command for training QM9:
```
python -m src.experimental.train --accelerator=gpu --devices=1 --num_workers=12 --dataset=qm9 --enable_wandb --wandb_run_id qm9_run --enable_progress_bar --check_samples_every_n_epoch 50
```

Command for training GEOM:
```
python -m src.experimental.train --accelerator=gpu --devices=1 --num_workers=12 --dataset=geom --enable_wandb --wandb_run_id geom_run --enable_progress_bar --check_samples_every_n_epoch 1 --batch_size 32 --max_epochs=100 --lr=2e-4
```

Running the same command with the same run_id will resume from the last checkpoint for that run_id.

## Evaluation

Setup for running baseline:
```
python scripts/eval/make_baseline_jobs.py
sbatch scripts/eval/run_baseline_jobs.py
```
This prepares and then submits an array of GNU parallel jobs, each of which run jobs that look like:
```
python -m src.experimental.evaluate_baseline --save_dir=where_to_save --split=test --enable_save_samples_and_examples --num_chunks=128 --chunk_id=0 --dataset=geom
```
The dataset is evaluated in multiple chunks for parallelism. Each job makes checkpoints and can continue from preemption by running the same command.


Scripts for evaluating the diffusion model:
- `scripts/eval/eval_p0.sh` - with all naturally abundant substitution coordinates
- `scripts/eval/eval_p10.sh` - with 10% dropout of substitution coordinates (QM9-C, GEOM-C)
- `scripts/eval/eval_rot_only.sh` - with no substitution coordinates, only moments
