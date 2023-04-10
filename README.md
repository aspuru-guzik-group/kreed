# unsigned-to-signed
Code for 3D structure determination by inferring signed from unsigned coordinates


Follow instructions in SETUP.md to setup the QM9 and GEOM datasets.


Setting up conda environment:
```
conda env create -f env.yml
```

Command for training QM9:
```
python -m src.experimental.train --accelerator=gpu --devices=1 --num_workers=12 --dataset=qm9 --run_id=qm9_run --enable_progress_bar --batch_size=512 --n_egnn_layers=6 --enable_wandb --n_visualize_samples=1 --evaluate_every_n_epochs=10 --n_sample_metric_batches=1 --max_epochs=3000 --lr=2e-4
```

Command for training GEOM:
```
python -m src.experimental.train --enable_progress_bar --n_sample_metric_batches=1 --num_workers=12 --timesteps=1000 --batch_size=64 --n_egnn_layers=5 --enable_wandb --equivariance=reflect --n_visualize_samples=1 --dataset=geom --run_id=geom_run
```

Running the same command with the same run_id will resume from the last checkpoint for that run_id.

