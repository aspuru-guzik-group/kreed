import collections

import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdchem import GetPeriodicTable

PTABLE = GetPeriodicTable()
ATOM_MASSES = torch.tensor([0] + [PTABLE.GetMostCommonIsotopeMass(z) for z in range(1, 119)], dtype=torch.float32)
ISOTOPICALLY_ABUNDANT_ATOMS = torch.tensor([5, 6, 7, 8, 14, 16, 17, 35, 80], dtype=torch.long)


def atom_masses_from_nums(atom_nums):
    return ATOM_MASSES.to(atom_nums.device)[atom_nums]


_Molecule = collections.namedtuple(
    "_Molecule",
    ["graph", "coords", "atom_nums", "cond_labels", "cond_mask", "moments", "id"],
)


class Molecule(_Molecule):

    @classmethod
    def from_dgl(cls, G):
        kwargs = {"graph": G}
        for field in _Molecule._fields:
            if field != "graph":
                kwargs[field] = G.ndata.pop(field)
        return _Molecule(**kwargs)

    @classmethod
    def to_dgl(cls, M):
        G = M.graph
        for field, val in M._asdict().items():
            if field != "graph":
                G.ndata[field] = val
        return G

    def xyzfile(self):
        file = f"{self.num_atoms}\n\n"
        for a, p in zip(self.atom_nums, self.xyz):
            x, y, z = p.tolist()
            file += f"{PTABLE.GetElementSymbol(int(a))} {x:f} {y:f} {z:f}\n"
        return file

    def smiles(self):
        mol = Chem.MolFromXYZBlock(self.xyzfile())
        rdDetermineBonds.DetermineConnectivity(mol)
        return Chem.MolToSmiles(mol)
