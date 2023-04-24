NAME=geom_run_reflect_eq_il
DIRECTORY=${NAME}_p0_samples
python scripts/generate_samples.py --directory=$DIRECTORY --checkpoint_path=checkpoints/$NAME/last.ckpt --dataset=geom --p_drop=0.0 --samples_per_example=3 --n_examples=$1 --split=test
python scripts/evaluate_samples.py --directory=$DIRECTORY --threshold=0.05 --k=3 > $DIRECTORY/eval.txt
