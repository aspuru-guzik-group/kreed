from src.datamodules import GEOMDatamodule
import torch

module = GEOMDatamodule(1)
G = next(iter(module.train_dataloader()))
exit()

def f(G_):
    return G_.local_var()

G2 = f(G)

G.ndata["xyz"] = torch.zeros_like(G.ndata["xyz"])
G.ndata["y"] = torch.zeros_like(G.ndata["xyz"])

print(G)
print(G2)
print(G2.ndata["xyz"][0])
print(G.ndata["xyz"][0])