import sys
sys.path.append('.')
import dgl
from src.evaluate import evaluate
import torch
import pickle
from tqdm import tqdm
from pathlib import Path
path = Path('v_geom_samples')

with open(path / 'truths.pkl', 'rb') as f:
    truths = pickle.load(f)

with open(path / 'reordered_flipped_samples.pkl', 'rb') as f:
    samples = pickle.load(f)

with open(path / 'trajs.pkl', 'rb') as f:
    trajs = pickle.load(f)


full_template_start = """<!DOCTYPE html>
<html>

<head>
    <style>
        .grid-container {
            display: grid;
            grid-template-columns: %s;
            grid-template-rows: %s;
            padding: 10px;
        }

        .grid-item {
            padding: 20px;
            font-size: 30px;
            text-align: center;
        }
    </style>
</head>

<body>
    <div class="grid-container">
"""

full_template_end = """
    </div>

</body>

</html>
"""

anim_grid_item_template = """<div class="grid-item">
            <div id="3dmolviewer%s" style="position: relative; width: 100%%; height: 100%%"></div>
            <script>
                var loadScriptAsync = function (uri) {
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

                if (typeof $3Dmolpromise === 'undefined') {
                    $3Dmolpromise = null;
                    $3Dmolpromise = loadScriptAsync('https://cdn.jsdelivr.net/npm/3dmol@latest/build/3Dmol-min.min.js');
                }

                var viewer = null;
                $3Dmolpromise.then(function () {
                    viewer = $3Dmol.createViewer(document.getElementById("3dmolviewer%s"), { backgroundColor: "white" });
                    viewer.zoomTo();
                    viewer.addModelsAsFrames(%s, "xyz");
                    viewer.setStyle({ "sphere": { "scale": 0.5 } });
                    viewer.animate({ "loop": "forward", "reps": 1, "interval": %s });
                    viewer.zoomTo();
                    viewer.render();
                });
            </script>

        </div>"""

mol_grid_item_template = """<div class="grid-item">
            <div id="3dmolviewer%s" style="position: relative; width: 100%%; height: 100%%"></div>
            <script>
                var loadScriptAsync = function (uri) {
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

                if (typeof $3Dmolpromise === 'undefined') {
                    $3Dmolpromise = null;
                    $3Dmolpromise = loadScriptAsync('https://cdn.jsdelivr.net/npm/3dmol@latest/build/3Dmol-min.min.js');
                }

                var viewer = null;
                $3Dmolpromise.then(function () {
                    viewer = $3Dmol.createViewer(document.getElementById("3dmolviewer%s"), { backgroundColor: "white" });
                    viewer.zoomTo();
                    viewer.addModel(%s, "xyz");
                    viewer.setStyle({ "sphere": { "scale": 0.5 } });
                    viewer.zoomTo();
                    viewer.render();
                });
            </script>

        </div>"""

from src.visualize.html import format_as_xyzfile

num_examples = 10
num_samples_per_example = 4

ground_truths = truths[:num_examples]

def to_xyzfile(G):
    atom_nums = G.ndata['atom_nums'].cpu().numpy()
    coords = G.ndata['xyz'].cpu().numpy()
    return format_as_xyzfile(atom_nums, coords)

def make_html_grid(ground_truths, samples, num_examples, num_samples_per_example, width=300, height=250):
    num_rows = num_samples_per_example+1
    num_cols = num_examples

    js = ''
    start = full_template_start % ("{}px ".format(width) * num_cols, "{}px ".format(height) * num_rows)
    js += start
    for i in range(num_rows):
        for j in range(num_cols):
            id = f"{i}_{j}"
            if i == 0:
                xyzfile = to_xyzfile(ground_truths[j])
                js += mol_grid_item_template % (id, id, repr(xyzfile))
            else:
                xyzfile = to_xyzfile(samples[j][i-1])
                js += mol_grid_item_template % (id, id, repr(xyzfile))
    js += full_template_end
    return js

width = 200
height = 200

mol_grid_js = make_html_grid(ground_truths, samples, num_examples, num_samples_per_example, width=width, height=height)
with open(path / 'mol_grid.html', 'w') as f:
    f.write(mol_grid_js)

from src.visualize.html import format_as_xyzfile

num_examples = 10
num_samples_per_example = 4

ground_truths = truths[:num_examples]

def make_anim_html_grid(ground_truths, samples, trajs, num_examples, num_samples_per_example, width=300, height=250):
    num_rows = num_samples_per_example+1
    num_cols = num_examples

    js = ''
    start = full_template_start % ("{}px ".format(width) * num_cols, "{}px ".format(height) * num_rows)
    js += start
    for i in range(num_rows):
        for j in range(num_cols):
            id = f"{i}_{j}"
            if i == 0:
                xyzfile = to_xyzfile(ground_truths[j])
                js += mol_grid_item_template % (id, id, repr(xyzfile))
            else:
                trajfile = ""
                atom_nums = samples[j][i-1].ndata['atom_nums'].cpu().numpy()
                # Prepend the last frame so zoomTo works
                trajfile += format_as_xyzfile(atom_nums, trajs[j][i-1][-1])
                for frame in trajs[j][i-1]:
                    trajfile += format_as_xyzfile(atom_nums, frame)
                js += anim_grid_item_template % (id, id, repr(trajfile), 100)
    js += full_template_end
    return js

width = 200
height = 200

anim_grid_js = make_anim_html_grid(ground_truths, samples, trajs, num_examples, num_samples_per_example, width=width, height=height)

with open(path / 'anim_grid.html', 'w') as f:
    f.write(anim_grid_js)
