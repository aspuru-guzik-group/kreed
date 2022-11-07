import rdkit
from rdkit.Chem import GetPeriodicTable
import torch

from src.datamodules.geom import GEOM_ATOMS

PTABLE = GetPeriodicTable()

def make_html(dgl_graph):
  Z = GEOM_ATOMS[dgl_graph.ndata['atom_nums']]
  XYZ = dgl_graph.ndata['xyz'].cpu().numpy()
  xyzfile = ""
  xyzfile += f"{len(Z)}\n\n"
  for atomic_num, xyz in zip(Z, XYZ):
      x, y, z = xyz
      xyzfile += f"{PTABLE.GetElementSymbol(int(atomic_num))} {x} {y} {z}\n"

  js = """<div id="3dmolviewer" style="position: relative; width: 100%%; height: 100%%"></div>

<script>
var loadScriptAsync = function(uri){
  return new Promise((resolve, reject) => {
    var tag = document.createElement('script');
    tag.src = uri;
    tag.async = true;
    tag.onload = () => {
      resolve();
    };
  var firstScriptTag = document.getElementsByTagName('script')[0];
  firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
});
};

if(typeof $3Dmolpromise === 'undefined') {
$3Dmolpromise = null;
  $3Dmolpromise = loadScriptAsync('https://cdn.jsdelivr.net/npm/3dmol@latest/build/3Dmol-min.min.js');
}

var viewer = null;
var warn = document.getElementById("3dmolwarning");
if(warn) {
    warn.parentNode.removeChild(warn);
}
$3Dmolpromise.then(function() {
viewer = $3Dmol.createViewer($("#3dmolviewer"),{backgroundColor:"white"});
viewer.zoomTo();
viewer.addModel(%s,"xyz");
viewer.setStyle({"sphere": {"scale": 0.5}});
viewer.zoomTo();
viewer.render();
});
</script>""" % repr(xyzfile)

  return js