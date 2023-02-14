import sys

sys.path.append('')
from src.diffusion.lit import LitEquivariantDDPM, LitEquivariantDDPMConfig
from src.experimental.train import TrainEquivariantDDPMConfig

model = LitEquivariantDDPM.load_from_checkpoint('logs/wandb/latest-run/files/last.ckpt')
from src.evaluate import evaluate
import dgl

from src.datamodules.qm9 import QM9Datamodule

from src.kraitchman import ATOM_MASSES

from tqdm import tqdm
import dgl

def get_com(G):
    with G.local_scope():
        m = ATOM_MASSES[G.ndata["atom_nums"].cpu()].to(G.device)
        G.ndata["m"] = m.unsqueeze(-1)
        com = dgl.sum_nodes(G, "xyz", weight="m") / dgl.sum_nodes(G, "m")
    return com

geom = QM9Datamodule(100, 1, zero_com=False)

samples_per_example = 10

dataset = geom.datasets['val']

N = len(dataset)

top_1_correctness = 0
top_1_C_rmsd_below_pt1 = 0

top_5_correctness = 0
top_5_C_rmsd_below_pt1 = 0

pbar = tqdm(dataset)
n_counted = 0
for G_true in pbar:
    n_counted += 1
    sample = model.edm.sample_p_G(dgl.batch([G_true for _ in range(samples_per_example)]))
    samples = dgl.unbatch(sample)
    coms = [get_com(G_pred) for G_pred in samples]

    atom_nums = G_true.ndata['atom_nums'].cpu().numpy()
    coords_true = G_true.ndata['xyz'].cpu().numpy()

    L2_coms = [com.square().sum() for com in coms]

    samples, L2_coms = zip(*sorted(zip(samples, L2_coms), key=lambda x: x[1]))

    results = []
    for G_pred in samples[:5]:
        result = evaluate(atom_nums, coords_true, G_pred.ndata['xyz'].cpu().numpy())
        results.append(result)
    
    abs_C_rmsds, C_rmsds, stabilities, correctnesses, heavy_rmsds = zip(*results)

    top_1_correctness += 1 if correctnesses[0] else 0
    top_1_C_rmsd_below_pt1 += 1 if C_rmsds[0] < 0.1 else 0

    top_5_correctness += 1 if any(correctnesses) else 0
    top_5_C_rmsd_below_pt1 += 1 if any([C_rmsd < 0.1 for C_rmsd in C_rmsds]) else 0

    pbar.set_description(f"Top 1 correctness: {top_1_correctness / n_counted}")


print(f"Top 1 correctness: {top_1_correctness / N}")
print(f"Top 1 C_rmsd < 0.1: {top_1_C_rmsd_below_pt1 / N}")

print(f"Top 5 correctness: {top_5_correctness / N}")
print(f"Top 5 C_rmsd < 0.1: {top_5_C_rmsd_below_pt1 / N}")
