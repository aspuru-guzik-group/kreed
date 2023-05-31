import collections
import functools

import dgl
import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdchem import GetPeriodicTable

PTABLE = GetPeriodicTable()
ATOM_MASSES = torch.tensor([0] + [PTABLE.GetMostCommonIsotopeMass(z) for z in range(1, 119)], dtype=torch.float32)
ISOTOPICALLY_ABUNDANT_ATOMS = torch.tensor([5, 6, 7, 8, 14, 16, 17, 35, 80], dtype=torch.long)


def atom_masses_from_nums(atom_nums):
    return ATOM_MASSES.to(atom_nums.device)[atom_nums]


_dgl_collate = dgl.dataloading.GraphCollator().collate

_Molecule = collections.namedtuple(
    "_Molecule",
    [
        "graph",
        "coords", "atom_nums", "masses", "masses_normalized",
        "cond_labels", "cond_mask",
        "moments", "id",
    ],
)


class Molecule(_Molecule):

    def replace(self, **kwargs):
        return self._replace(**kwargs)

    # =============
    # DGL Utilities
    # =============

    @classmethod
    def from_dgl(cls, G):
        kwargs = {"graph": G}
        for field in Molecule._fields:
            if field != "graph":
                kwargs[field] = G.ndata.pop(field)
        return Molecule(**kwargs)

    @classmethod
    def to_dgl(cls, M):
        G = M.graph
        for field, val in M._asdict().items():
            if field != "graph":
                G.ndata[field] = val
        return G

    def broadcast(self, x):
        return dgl.broadcast_nodes(self.graph, x)

    def readout_pool(self, x, op, broadcast=False):
        self.graph.ndata["tmp"] = x.flatten(start_dim=1) if (x.ndim > 1) else x.unsqueeze(-1)
        pooled = dgl.readout_nodes(self.graph, "tmp", op=op)
        del self.graph.ndata["tmp"]
        if broadcast:
            pooled = self.broadcast(pooled)
        return pooled.view(-1, *x.shape[1:])

    sum_pool = functools.partialmethod(readout_pool, op="sum")
    mean_pool = functools.partialmethod(readout_pool, op="mean")

    # ==============
    # Chem Utilities
    # ==============

    def xyzfile(self):
        assert not self.batched
        file = f"{self.coords.shape[0]}\n\n"
        for a, p in zip(self.atom_nums, self.coords):
            x, y, z = p.tolist()
            file += f"{PTABLE.GetElementSymbol(int(a))} {x:f} {y:f} {z:f}\n"
        return file

    def smiles(self):
        assert not self.batched
        mol = Chem.MolFromXYZBlock(self.xyzfile())
        rdDetermineBonds.DetermineConnectivity(mol)
        return Chem.MolToSmiles(mol)

    # ==========
    # Properties
    # ==========

    @classmethod
    def collate(cls, molecules):
        G = _dgl_collate([Molecule.to_dgl(M) for M in molecules])
        return Molecule.from_dgl(G)

    def unbatch(self):
        return [Molecule.from_dgl(G) for G in dgl.unbatch(Molecule.to_dgl(self))]

    @property
    def batched(self):
        return self.batch_size > 1

    @property
    def batch_size(self):
        return self.graph.batch_size

    @property
    def num_atoms(self):
        return self.graph.batch_num_nodes()

    @property
    def device(self):
        return self.graph.device

    def cpu(self):
        return self._replace(**{field: x.cpu() for field, x in self._asdict().items()})

    @property
    def id_as_int(self):
        assert not self.batched
        return int(self.id[0][0])
