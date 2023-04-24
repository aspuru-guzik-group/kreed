NAME=qm9_run_main
DIRECTORY=${NAME}_p10_samples
python scripts/generate_samples.py --directory=$DIRECTORY --checkpoint_path=checkpoints/$NAME/last.ckpt --dataset=qm9 --p_drop=0.10 --samples_per_example=3 --n_examples=$1 --split=test
python scripts/evaluate_samples.py --directory=$DIRECTORY --threshold=0.05 --k=3 > $DIRECTORY/eval.txt
