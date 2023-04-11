import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdchem import GetPeriodicTable

PTABLE = GetPeriodicTable()
ATOM_MASSES = torch.tensor([0] + [PTABLE.GetMostCommonIsotopeMass(z) for z in range(1, 119)], dtype=torch.float32)
ISOTOPICALLY_ABUNDANT_ATOMS = torch.tensor([5, 6, 7, 8, 14, 16, 17, 35, 80], dtype=torch.long)

def _build_atom_map_cache():
    geom_atoms = torch.tensor([1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83], dtype=torch.long)

    ztoi = torch.full([84], -100, dtype=torch.long)
    for i, z in enumerate(geom_atoms):
        ztoi[z] = i
    return ztoi


ATOM_ZTOI = _build_atom_map_cache()
# _Molecule = collections.namedtuple(
#     "Molecule",
#     ["xyz", "atom_nums", "atom_masses", "cond_labels", "mask", "cond_mask", "moments", "id", "num_atoms"],
# )


# class Molecule(_Molecule):

#     @property
#     def batched(self):
#         return self.xyz.ndim > 2

#     @property
#     def batch_size(self):
#         return self.xyz.shape[0] if self.batched else None

#     @property
#     def device(self):
#         return self.xyz.device

#     def cpu(self):
#         return self._replace(**{field: x.cpu() for field, x in self._asdict().items()})

#     def unbatch(self):
#         assert self.batched
#         return [
#             Molecule(
#                 xyz=self.xyz[i, :n],
#                 atom_nums=self.atom_nums[i, :n],
#                 atom_masses=self.atom_masses[i, :n],
#                 cond_labels=self.cond_labels[i, :n],
#                 mask=None,
#                 cond_mask=self.cond_mask[i, :n],
#                 moments=self.moments[i],
#                 id=self.id[i].item(),
#                 num_atoms=self.num_atoms[i].item(),
#             )
#             for i, n in zip(range(self.batch_size), self.num_atoms)
#         ]

#     def xyzfile(self):
#         assert not self.batched
#         file = f"{self.num_atoms}\n\n"
#         for a, p in zip(self.atom_nums, self.xyz):
#             x, y, z = p.tolist()
#             file += f"{PTABLE.GetElementSymbol(int(a))} {x:f} {y:f} {z:f}\n"
#         return file

#     def smiles(self):
#         mol = Chem.MolFromXYZBlock(self.xyzfile())
#         rdDetermineBonds.DetermineConnectivity(mol)
#         return Chem.MolToSmiles(mol)
