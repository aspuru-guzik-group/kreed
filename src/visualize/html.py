import pathlib

from rdkit.Chem import GetPeriodicTable

PTABLE = GetPeriodicTable()

ROOT = pathlib.Path(__file__).parent
with open(ROOT / "molecule_template.html", "r") as f:
    JS_MOLECULE_TEMPLATE = f.read()
with open(ROOT / "trajectory_template.html", "r") as f:
    JS_TRAJECTORY_TEMPLATE = f.read()


def html_render_molecule(M):
    return JS_MOLECULE_TEMPLATE % (M.id, repr(M.xyzfile()))


def html_render_trajectory(Ms):
    assert isinstance(Ms, list)

    trajfile = ""
    for M in Ms:
        trajfile += M.xyzfile()
    trajfile = Ms[-1].xyzfile() + trajfile  # prepend the last frame so zoomTo works
    return JS_TRAJECTORY_TEMPLATE % (Ms[0].id, repr(trajfile))