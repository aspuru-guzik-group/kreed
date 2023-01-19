import pathlib

from rdkit.Chem import GetPeriodicTable

PTABLE = GetPeriodicTable()

ROOT = pathlib.Path(__file__).parent
with open(ROOT / "molecule_template.html", "r") as f:
    JS_MOLECULE_TEMPLATE = f.read()
with open(ROOT / "trajectory_template.html", "r") as f:
    JS_TRAJECTORY_TEMPLATE = f.read()


def format_as_xyzfile(atom_nums, coords):
    xyzfile = f"{atom_nums.shape[0]}\n\n"
    for a, xyz in zip(atom_nums, coords):
        x, y, z = xyz
        xyzfile += f"{PTABLE.GetElementSymbol(int(a))} {x} {y} {z}\n"
    return xyzfile


def html_render_molecule(geom_id, atom_nums, coords):
    xyzfile = format_as_xyzfile(atom_nums, coords)
    return JS_MOLECULE_TEMPLATE % (geom_id, repr(xyzfile))


def html_render_trajectory(geom_id, atom_nums, coords_trajectory):
    assert isinstance(coords_trajectory, list)

    trajfile = ""
    for coords in coords_trajectory:
        trajfile += format_as_xyzfile(atom_nums, coords)

    # Prepend the last frame so zoomTo works
    trajfile = format_as_xyzfile(atom_nums, coords_trajectory[-1]) + trajfile

    return JS_TRAJECTORY_TEMPLATE % (geom_id, repr(trajfile))
