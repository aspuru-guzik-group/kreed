ROOT=/local-scratch/geom_chunks
CKPT=final_checkpoints/geom.ckpt
NSAMPLES=10
K=5
THRESHOLD=0.05
p=0.0
DIRECTORY=$ROOT/p${p}

# for chunk in 0 1 2 3 4 
for chunk in 5 6 7 8 9
do
    python scripts/eval/generate_eval_chunk_geom_samples.py --directory=$DIRECTORY --checkpoint_path=$CKPT --p_drop=$p --samples_per_example=$NSAMPLES --split=test --k=$K --threshold=$THRESHOLD --chunk=$chunk
done
