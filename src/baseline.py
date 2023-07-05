import sys
sys.path.append('.')

from rdkit.Chem import GetPeriodicTable
PTABLE = GetPeriodicTable()

def to_xyzfile(atom_nums, coords):
    file = f"{coords.shape[0]}\n\n"
    for a, p in zip(atom_nums, coords):
        x, y, z = p.tolist()
        file += f"{PTABLE.GetElementSymbol(int(a))} {x:f} {y:f} {z:f}\n"
    return file

from src.datamodule import ConformerDatamodule
import torch
from src.chem import Molecule
import numpy as np

import tqdm
import warnings

from src.datamodule import ConformerDatamodule
import os

def get_dists(coords):
    dists = torch.triu(torch.sum((coords.reshape(-1, 1, 3)-coords.reshape(1, -1, 3))**2, axis=-1))
    dists = dists[dists != 0].sqrt()
    return dists.numpy()

def get_heavy_coords(M):
    heavy_mask = (M.atom_nums != 1).squeeze(-1)
    coords = M.coords[heavy_mask]
    return coords


if os.path.exists('.qm9_train_dists.npy'):
    qm9_train_dists = np.load('.qm9_train_dists.npy')
    geom_train_dists = np.load('.geom_train_dists.npy')
else:
    data = ConformerDatamodule(
        dataset='qm9',
        seed=100,
        batch_size=1,
        split_ratio=(0.8, 0.1, 0.1),
        num_workers=0,
        distributed=False,
        tol=-1.0,
    )
    qm9_train_dists = []
    for i in tqdm.tqdm(range(len(data.datasets['train']))):
        if i not in {76327, 57089, 41257, 64665}: # bad examples
            dists = get_dists(get_heavy_coords(data.datasets['train'][i]))
            qm9_train_dists.extend(dists)
    qm9_train_dists = np.array(qm9_train_dists)
    np.save('.qm9_train_dists.npy', qm9_train_dists)

    data = ConformerDatamodule(
        dataset='geom',
        seed=100,
        batch_size=1,
        split_ratio=(0.8, 0.1, 0.1),
        num_workers=0,
        distributed=False,
        tol=-1.0,
        only_lowest_energy_conformers=True,
    )
    geom_train_dists = []
    for i in tqdm.tqdm(range(len(data.datasets['train']))):
        dists = get_dists(get_heavy_coords(data.datasets['train'][i]))
        geom_train_dists.extend(dists)
    geom_train_dists = np.array(geom_train_dists)
    np.save('.geom_train_dists.npy', geom_train_dists)

N = 60000
rightbound = 200
left_bins = np.linspace(0, rightbound, N)
qm9_values, bin_edges = np.histogram(qm9_train_dists, bins=left_bins, density=True)
geom_values, bin_edges = np.histogram(geom_train_dists, bins=left_bins, density=True)
values = {'qm9': qm9_values, 'geom': geom_values}

def scoring(dists, dataset='qm9'):
    idx = (dists * N / rightbound).astype(int)
    if np.any(idx > N):
        return 1e6
    vals = values[dataset][idx]
    if 0 in vals:
        return 1e6
    return -np.log(vals).sum()

import numpy as np
import deap
from deap import algorithms, base, tools, creator

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
import biotite.structure.io.mol as iomol
import io
import hydride

# based on https://stackoverflow.com/questions/31325025/how-to-perform-discrete-optimization-of-functions-over-matrices

class Individual(object):

    def __init__(self, discrete, continuous):
        self.discrete = discrete
        self.continuous = continuous

class GeneticMinimizer(object):

    def __init__(self, dataset, masses, moments, cond_mask, cond_labels, popsize=500, seed=0, verbose=True, num_samples=10):
        self.verbose = verbose

        self.dataset = dataset
        self.masses = masses
        self.moments = moments
        self.cond_mask = cond_mask
        self.flat_cond_mask = cond_mask.flatten()
        self.cond_labels = cond_labels
        self.N = len(masses)
        
        # an 'individual' consists of a (D,) flat numpy array of 0s and 1s
        # and a (U,) flat array of continuous values for coordinates with unknown substitution coordinates

        self.U = np.sum(~self.flat_cond_mask)
        self.D = np.sum(self.flat_cond_mask)

        self._gen = np.random.RandomState(seed)

        self._cr = creator
        self._cr.create("FitnessMin", base.Fitness, weights=(-1.0,))
        self._cr.create("Individual", Individual, fitness=self._cr.FitnessMin)

        self._tb = base.Toolbox()
        
        def create_indiv():
            discrete = self._gen.randint(2, size=self.D)
            continuous = self._gen.randn(self.U)
            return self._cr.Individual(discrete, continuous)
        
        self._tb.register("individual", create_indiv)

        # the 'population' consists of a list of such individuals
        self._tb.register("population", tools.initRepeat, list,
                          self._tb.individual)
        self._tb.register("evaluate", self.fitness)
        self._tb.register("mate", self.crossover)
        self._tb.register("mutate", self.mutate, rate=0.02)
        self._tb.register("select", tools.selTournament, tournsize=3)

        # create an initial population, and initialize a hall-of-fame to store
        # the best individual
        self.pop = self._tb.population(n=popsize)
        def similar(a, b):
            return np.array_equal(a.discrete, b.discrete)
        self.hof = tools.HallOfFame(num_samples, similar=similar)

        # print summary statistics for the population on each iteration
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        # keep track of configurations that have already been visited
        self.tabu = set()

    def random_bool(self, *args):
        return self._gen.rand(*args) < 0.5

    def mutate(self, ind, rate=1E-3):
        """
        mutate an individual by bit-flipping one or more randomly chosen
        elements
        """

        tries = 0
        # ensure that each mutation always introduces a novel configuration
        while np.packbits(ind.discrete.astype(np.uint8)).tobytes() in self.tabu:
            if tries >= min(self.D, 50):
                break
            
            n_flip = self._gen.binomial(self.D, rate)
            if not n_flip:
                continue
            idx = self._gen.randint(0, self.D, n_flip)
            ind.discrete[idx] = ~ind.discrete[idx]
            tries += 1
        
        ind.continuous += self._gen.randn(self.U)

        return ind,

    def fitness(self, individual):
        """
        assigns a fitness value to each individual, based on the determinant
        """
        h = np.packbits(individual.discrete.astype(np.uint8)).tobytes()
        self.tabu.add(h)

        coords = np.zeros((self.N, 3))
        signs = individual.discrete * 2 - 1
        coords[self.cond_mask] = signs * self.cond_labels[self.cond_mask]
        coords[~self.cond_mask] = individual.continuous
        
        dyadic = self.masses.reshape(self.N, 1, 1) * coords.reshape(self.N, 3, 1) * coords.reshape(self.N, 1, 3) # (N 1 1) * (N 3 1) * (N 1 3)
        dyadic = np.sum(dyadic, axis=0)  # (3 3)

        diag = np.diag(self.moments)
        moment_error = np.linalg.norm(np.triu(dyadic - diag))

        cm = np.mean(coords * self.masses.reshape(-1, 1), axis=0)
        cm_error = np.linalg.norm(cm)

        dists = np.triu(np.sum((coords.reshape(-1, 1, 3)-coords.reshape(1, -1, 3))**2, axis=-1))
        dists = np.sqrt(dists[dists != 0])
        dist_error = scoring(dists, self.dataset)

        fitness = moment_error + cm_error + dist_error

        return fitness,

    def crossover(self, ind1, ind2):
        """
        swaps a random swath between two individuals
        """
        
        col = self._gen.randint(0, 3)
        skip = self._gen.randint(3, 6)

        ind1.discrete[col::skip], ind2.discrete[col::skip] = (
            ind2.discrete[col::skip].copy(), ind1.discrete[col::skip].copy())
        
        if self.D > 1:
            cx1 = self._gen.randint(0, self.D - 1)
            cx2 = self._gen.randint(cx1, self.D)

            ind1.discrete[cx1:cx2], ind2.discrete[cx1:cx2] = (
                ind2.discrete[cx1:cx2].copy(), ind1.discrete[cx1:cx2].copy())
        
        return ind1, ind2

    def run(self, ngen=int(1E6), mutation_rate=0.3, crossover_rate=0.7):

        pop, log = algorithms.eaSimple(self.pop, self._tb,
                                       cxpb=crossover_rate,
                                       mutpb=mutation_rate,
                                       ngen=ngen,
                                       stats=self.stats,
                                       halloffame=self.hof,
                                       verbose=self.verbose)
        self.log = log

        return self.hof, log


def cm_score(atom_nums, coords):
    masses = np.array([PTABLE.GetMostCommonIsotopeMass(int(atom)) for atom in atom_nums.squeeze(-1)])
    cm = np.mean(coords * masses.reshape(-1, 1), axis=0)
    cm_error = np.linalg.norm(cm)

    dists = np.sqrt(np.sum((coords.reshape(-1, 1, 3)-coords.reshape(1, -1, 3))**2, axis=-1))
    dists += np.eye(len(coords)) * 1e6
    min_dist = np.min(dists)
    dist_error = (min_dist < 1.09) * (1.09-min_dist)**2*1000
    return cm_error + dist_error

def run_baseline(M, dataset, num_samples=10, ngen=10, popsize=20000, verbose=True):
    if not verbose:
        warnings.filterwarnings("ignore")

    # prepare input for heavy atom framework optimization
    atom_nums = M.atom_nums.numpy()
    masses = M.masses.numpy()
    moments = M.moments[0].numpy()
    cond_mask = M.cond_mask.numpy()
    cond_labels = M.cond_labels.numpy()

    # restrict to heavy atoms
    heavy_mask = (atom_nums != 1).squeeze()
    atom_nums = atom_nums[heavy_mask]
    masses = masses[heavy_mask]
    cond_mask = cond_mask[heavy_mask]
    cond_labels = cond_labels[heavy_mask]

    # optimize heavy atom framework
    gd = GeneticMinimizer(dataset, masses=masses, moments=moments, cond_labels=cond_labels, cond_mask=cond_mask, popsize=popsize, verbose=verbose, num_samples=num_samples)
    bests, log = gd.run(ngen=ngen,
                        mutation_rate=0.7,
                        crossover_rate=0.9,
                        )

    # post process generated frameworks
    all_candidates = []
    for best in bests:
        coords = np.zeros((gd.N, 3))
        signs = best.discrete * 2 - 1
        coords[gd.cond_mask] = signs * gd.cond_labels[gd.cond_mask]
        coords[~gd.cond_mask] = best.continuous

        # add hydrogens
        original_num_H = (M.atom_nums.squeeze(-1) == 1).sum().item()

        file = to_xyzfile(atom_nums, coords)
        mol = Chem.MolFromXYZBlock(file)
        rdDetermineBonds.DetermineConnectivity(mol)
        molecule = iomol.MOLFile.read(io.StringIO(Chem.MolToMolBlock(mol))).get_structure()
        molecule, mask = hydride.add_hydrogen(molecule)

        final_num_H = np.sum(~mask)
        extra_H = final_num_H - original_num_H

        original_h_added_coords = np.array(molecule.coord)
        original_h_added_atom_nums = np.array([PTABLE.GetAtomicNumber(atom.capitalize()) for atom in molecule.element])[:, None]

        h_added_candidates = []
        for _ in range(1000):
            h_added_coords = original_h_added_coords.copy()
            h_added_atom_nums = original_h_added_atom_nums.copy()
            if extra_H > 0:

                choices = np.nonzero(~mask)[0]
                np.random.shuffle(choices)
                drop_idxs = choices[:extra_H]

                h_added_coords = np.delete(h_added_coords, drop_idxs, axis=0)
                h_added_atom_nums = np.delete(h_added_atom_nums, drop_idxs, axis=0)
            elif extra_H < 0:

                h_added_coords = np.concatenate([h_added_coords, np.random.randn(abs(extra_H), 3)], axis=0)
                h_added_atom_nums = np.concatenate([h_added_atom_nums, np.ones((abs(extra_H), 1),)], axis=0)
            h_added_candidates.append((h_added_atom_nums, h_added_coords))
        
        final_atom_nums, final_coords = min(h_added_candidates, key=lambda x: cm_score(*x))

        sorts = np.argsort(final_atom_nums.squeeze(-1))
        unsorts = np.argsort(np.argsort(M.atom_nums.numpy().squeeze(-1)))
        final_atom_nums = final_atom_nums.squeeze(-1)[sorts][unsorts].reshape(-1, 1)
        final_coords = final_coords[sorts][unsorts]

        all_candidates.append(
            Molecule(
                graph=None,
                atom_nums=torch.tensor(final_atom_nums).to(M.atom_nums),
                coords=torch.tensor(final_coords).to(M.coords),
                masses=M.masses,
                masses_normalized=M.masses_normalized,
                moments=M.moments,
                cond_labels=M.cond_labels,
                cond_mask=M.cond_mask,
                id=M.id,
            )
        )
    
    return all_candidates

if __name__ == '__main__':

    data = ConformerDatamodule(
        dataset='geom',
        seed=100,
        batch_size=1,
        split_ratio=(0.8, 0.1, 0.1),
        num_workers=0,
        distributed=False,
        tol=-1.0,
        only_lowest_energy_conformers=True
    )

    dataset = data.datasets['train']

    # M = dataset[123]
    M = dataset[486]

    # start timer
    import time
    start = time.time()
    all_candidates = run_baseline(M, 'geom', ngen=50, num_samples=10, popsize=20000, verbose=True)
    print(f'Elapsed time: {time.time()-start:.2f} s')

    import pickle
    with open('genetic_minimization.pkl', 'wb') as f:
        pickle.dump(all_candidates, f)


