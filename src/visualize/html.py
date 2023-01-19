PTABLE = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 33: 'As', 35: 'Br', 53: 'I', 80: 'Hg', 83: 'Bi'
}

with open("js_template.txt", "r") as f:
    JS_TEMPLATE = f.read()
with open("js_anim_template.txt", "r") as f:
    JS_ANIM_TEMPLATE = f.read()


def to_xyzfile(atom_nums, coords):
    xyzfile = f"{atom_nums.shape[0]}\n\n"
    for a, xyz in zip(atom_nums, coords):
        x, y, z = xyz
        xyzfile += f"{PTABLE[int(a)]} {x} {y} {z}\n"
    return xyzfile


def html_render(geom_id, atom_nums, coords):
    xyzfile = to_xyzfile(atom_nums, coords)
    return JS_TEMPLATE % (geom_id, repr(xyzfile))


def html_render_animate(geom_id, atom_nums_list, coords_list):
    assert type(atom_nums_list) is list and type(coords_list) is list

    trajfile = ""
    for atom_nums, coords in zip(atom_nums_list, coords_list):
        trajfile += to_xyzfile(atom_nums, coords)

    # Prepend the last frame so zoomTo works
    trajfile = to_xyzfile(atom_nums_list[-1], coords_list[-1]) + trajfile

    return JS_ANIM_TEMPLATE % (geom_id, repr(trajfile))
