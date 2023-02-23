
import py3Dmol
from .html import format_as_xyzfile

def show(xyzfile):
    view = py3Dmol.view(width=400, height=400)
    view.addModel(xyzfile, 'xyz')
    view.setStyle({'sphere':{'scale': .5}})
    # view.setBackgroundColor('0xeeeeee')
    view.zoomTo()
    view.show()

def show_graph(G):
    atom_nums = G.ndata['atom_nums'].cpu().numpy()
    coords_true = G.ndata['xyz'].cpu().numpy()

    show(format_as_xyzfile(atom_nums, coords_true))

def show_xyz_atoms(atom_nums, coords):
    show(format_as_xyzfile(atom_nums, coords))

