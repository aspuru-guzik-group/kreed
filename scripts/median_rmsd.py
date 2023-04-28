import pickle
from pathlib import Path
# parameters
k = 3

for directory in ['qm9_run_main_p0_samples', 'qm9_run_main_p10_samples', 'geom_run_reflect_eq_il_p0_samples', 'geom_run_reflect_eq_il_p10_samples']:

    print(directory)
    path = Path(directory)
    with open(path / 'all_results.pkl', 'rb') as f:
        all_results = pickle.load(f)

    top_1_rmsds = []
    top_k_rmsds = []
    for result in all_results:
        top_1_rmsds.append(result[0]['heavy_coord_rmse'])
        top_k_rmsds.append(min([r['heavy_coord_rmse'] for r in result[:k]]))

    # median of top_1_rmsds
    import numpy as np
    print("Top 1 median RMSD:", np.median(top_1_rmsds))
    print(f"Top {k} median RMSD:", np.median(top_k_rmsds))
    print()
