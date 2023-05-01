# unsigned-to-signed
Code for 3D structure determination by inferring signed from unsigned coordinates


Follow instructions in SETUP.md to setup the QM9 and GEOM datasets.


Setting up conda environment:
```
conda env create -f env.yml
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

