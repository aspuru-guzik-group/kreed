from src.xyz2mol import xyz2mol

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import spyrmsd.rmsd
import numpy as np

from copy import deepcopy


def evaluate(atom_nums, coords_true, coords_pred):
    carbon_mask = (atom_nums == 6)
    C_coords_true = coords_true[carbon_mask]
    C_coords_pred = coords_pred[carbon_mask]
    C_atomic_nums = atom_nums[carbon_mask]

    abs_C_rmsd = np.sqrt(np.mean((np.abs(C_coords_true) - np.abs(C_coords_pred))**2))
    
    # take min(left, right) enantiomer
    rmsd1 = spyrmsd.rmsd.rmsd(
        coords1=C_coords_true,
        coords2=C_coords_pred,
        atomicn1=C_atomic_nums,
        atomicn2=C_atomic_nums,
        minimize=True
    )

    C_coords_pred[:, 0] *= -1  # flip x coordinates for other enantiomer

    rmsd2 = spyrmsd.rmsd.rmsd(
        coords1=C_coords_true,
        coords2=C_coords_pred,
        atomicn1=C_atomic_nums,
        atomicn2=C_atomic_nums,
        minimize=True
    )

    C_rmsd = min(rmsd1, rmsd2)
    
    if C_rmsd < 15:
        try:
            pred_mols = xyz2mol(
                atoms=atom_nums.tolist(),
                coordinates=coords_pred.tolist(),
                embed_chiral=False,
            )
        except ValueError:
            pred_mols = False
    else:
        pred_mols = False

    stable = True if pred_mols else False

    correct = False

    if pred_mols:
        try:
            true_mols = xyz2mol(
                atoms=atom_nums.tolist(),
                coordinates=coords_true.tolist(),
                embed_chiral=False,
            )
        except ValueError:
            true_mols = False
        
        if true_mols:
            true_smiles = Chem.CanonSmiles(Chem.MolToSmiles(true_mols[0]))
            pred_smiles = Chem.CanonSmiles(Chem.MolToSmiles(pred_mols[0]))
            correct = True if true_smiles == pred_smiles else False
    
    heavy_rmsd = None
    if correct:
        # get rmsd from mol

        spyrmsd_mol_true = spyrmsd.molecule.Molecule.from_rdkit(true_mols[0])
        spyrmsd_mol_pred = spyrmsd.molecule.Molecule.from_rdkit(pred_mols[0])
        flipped = deepcopy(spyrmsd_mol_pred)
        flipped.coordinates[:, 0] *= -1 # flip x coordinates for other enantiomer

        rmsds = spyrmsd.rmsd.rmsdwrapper(
                spyrmsd_mol_true,
                [spyrmsd_mol_pred, flipped],
                symmetry=True,
                minimize=True,
            )
        heavy_rmsd = min(rmsds)

    atom_nums_no_H = atom_nums[atom_nums != 1]
    coords_true_no_H = coords_true[atom_nums != 1]
    coords_pred_no_H = coords_pred[atom_nums != 1]
    
    if C_rmsd < 15:
        try:
            pred_mols = xyz2mol(
                atoms=atom_nums_no_H.tolist(),
                coordinates=coords_pred_no_H.tolist(),
                embed_chiral=False,
            )
        except ValueError:
            pred_mols = False
    else:
        pred_mols = False

    stable_no_H = True if pred_mols else False

    correct_no_H = False

    if pred_mols:
        try:
            true_mols = xyz2mol(
                atoms=atom_nums_no_H.tolist(),
                coordinates=coords_true_no_H.tolist(),
                embed_chiral=False,
            )
        except ValueError:
            true_mols = False
        
        if true_mols:
            true_smiles = Chem.CanonSmiles(Chem.MolToSmiles(true_mols[0]))
            pred_smiles = Chem.CanonSmiles(Chem.MolToSmiles(pred_mols[0]))
            correct_no_H = True if true_smiles == pred_smiles else False
    
    heavy_rmsd_no_H = None
    if correct_no_H:
        # get rmsd from mol

        spyrmsd_mol_true = spyrmsd.molecule.Molecule.from_rdkit(true_mols[0])
        spyrmsd_mol_pred = spyrmsd.molecule.Molecule.from_rdkit(pred_mols[0])
        flipped = deepcopy(spyrmsd_mol_pred)
        flipped.coordinates[:, 0] *= -1 # flip x coordinates for other enantiomer

        rmsds = spyrmsd.rmsd.rmsdwrapper(
                spyrmsd_mol_true,
                [spyrmsd_mol_pred, flipped],
                symmetry=True,
                minimize=True,
            )
        heavy_rmsd_no_H = min(rmsds)

    return abs_C_rmsd, C_rmsd, (stable_no_H or stable), (correct_no_H or correct), (heavy_rmsd_no_H or heavy_rmsd)
