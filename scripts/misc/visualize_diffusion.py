import dgl
import wandb

from src.datamodules import QM9Datamodule
from src.visualize import html_render_molecule


def visualize_diffusion():
    wandb.init(project="visualize_qm9_diffusion")

    qm9 = QM9Datamodule(seed=0, batch_size=6)
    G_batch = next(iter(qm9.train_dataloader()))

    for i, G in enumerate(dgl.unbatch(G_batch)):
        geom_id = G.ndata["id"][0].item()
        atom_nums = G.ndata["atom_nums"].cpu().numpy()
        coords = G.ndata["xyz"].cpu().numpy()

        render = html_render_molecule(geom_id, atom_nums, coords)
        wandb.log({f"mol/{i}": wandb.Html(render)})


if __name__ == "__main__":
    visualize_diffusion()
