
PTABLE = {
  1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 33: 'As', 35: 'Br', 53: 'I', 80: 'Hg', 83: 'Bi'
}

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

def to_xyz(atom_nums, coords):
    xyzfile = f"{atom_nums.shape[0]}\n\n"
    for a, xyz in zip(atom_nums, coords):
        x, y, z = xyz
        xyzfile += f"{PTABLE[int(a)]} {x} {y} {z}\n"
    return xyzfile

def html_render(geom_id, atom_nums, coords):
    xyzfile = to_xyz(atom_nums, coords)
    return JS_TEMPLATE % (geom_id, repr(xyzfile))

JS_ANIM_TEMPLATE = (
"""<div><p>GEOM ID: %d</p></div>
<div id="3dmolviewer" style="position: relative; width: 100%%; height: 100%%"></div>
<script>

var loadScriptAsync = function(uri){
  return new Promise((resolve, reject) => {
    //this is to ignore the existence of requirejs amd
    var savedexports, savedmodule;
    if (typeof exports !== 'undefined') savedexports = exports;
    else exports = {}
    if (typeof module !== 'undefined') savedmodule = module;
    else module = {}

    var tag = document.createElement('script');
    tag.src = uri;
    tag.async = true;
    tag.onload = () => {
        exports = savedexports;
        module = savedmodule;
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
$3Dmolpromise.then(function() {
viewer = $3Dmol.createViewer(document.getElementById("3dmolviewer"),{backgroundColor:"white"});
viewer.zoomTo();
    viewer.addModelsAsFrames(%s,"xyz");
    viewer.setStyle({"sphere": {"scale": 0.5}});
    viewer.animate({"loop": "forward", "interval": 150});
    viewer.zoomTo();
viewer.render();
});
</script>"""
)

def html_render_animate(geom_id, atom_nums_list, coords_list):

    assert type(atom_nums_list) is list and type(coords_list) is list
    
    trajfile = ''
    for atom_nums, coords in zip(atom_nums_list, coords_list):
        trajfile += to_xyz(atom_nums, coords)
    
    # prepend the last frame so zoomTo works
    trajfile = to_xyz(atom_nums_list[-1], coords_list[-1]) + trajfile

    return JS_ANIM_TEMPLATE % (geom_id, repr(trajfile))
