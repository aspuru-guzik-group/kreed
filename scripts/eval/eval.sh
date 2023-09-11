num_chunks=50
name=final_rtx_huge_uniform_300
split=val
batch_size=100
python -m src.experimental.evaluate --num_workers=12 --chunk_id=0 --num_chunks=$num_chunks --checkpoint_path=final_checkpoints/$name.ckpt --save_dir=/local-scratch/$name/0.1 --enable_save_samples_and_examples --enable_only_carbon_cond --pdropout_cond 0.1 0.1 --split=$split --batch_size=$batch_size
python -m src.experimental.evaluate --num_workers=12 --chunk_id=0 --num_chunks=$num_chunks --checkpoint_path=final_checkpoints/$name.ckpt --save_dir=/local-scratch/$name/0.0 --enable_save_samples_and_examples --split=$split --batch_size=$batch_size
