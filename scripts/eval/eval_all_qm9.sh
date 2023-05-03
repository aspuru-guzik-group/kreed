ROOT=/local-scratch/qm9_run_final_samples
CKPT=final_checkpoints/qm9.ckpt
NSAMPLES=10
K=5
THRESHOLD=0.05

for p in 0.0 0.1
do
    DIRECTORY=$ROOT/p${p}
    python scripts/eval/generate_all_qm9_samples.py --directory=$DIRECTORY --checkpoint_path=$CKPT --p_drop=$p --samples_per_example=$NSAMPLES --split=test
    python scripts/eval/evaluate_all_qm9_samples.py --directory=$DIRECTORY --threshold=$THRESHOLD --k=$K > $DIRECTORY/eval.txt
done
