from rdkit.Chem import GetPeriodicTable

PTABLE = GetPeriodicTable()

JS_TEMPLATE = (
"""<div><p>GEOM ID: %d</p></div>
<div id="3dmolviewer" style="position: relative; width: 100%%; height: 100%%"></div>

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
</script>"""
)


def html_render(geom_id, atom_nums, coords):
    xyzfile = f"{atom_nums.shape[0]}\n\n"
    for a, xyz in zip(atom_nums, coords):
        x, y, z = xyz
        xyzfile += f"{PTABLE.GetElementSymbol(int(a))} {x} {y} {z}\n"
    return JS_TEMPLATE % (geom_id, repr(xyzfile))
