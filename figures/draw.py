import py3Dmol
import torch

def add(view, start, end, obj='arrow', viewer=(0,0), **kwargs):
    if obj == 'arrow':
        view.addArrow({
            "start": {k: float(v) for k, v in zip("xyz", start)},
            "end": {k: float(v) for k, v in zip("xyz", end)},
            **kwargs,
        }, viewer=viewer)
    elif obj == 'cylinder':
        view.addCylinder({
            "start": {k: float(v) for k, v in zip("xyz", start)},
            "end": {k: float(v) for k, v in zip("xyz", end)},
            **kwargs,
        }, viewer=viewer)
    return view

def compare_fit(view, M_true, M_pred, length=.65, radius=0.01, color='white', obj="cylinder", radiusRatio=1, cap=1, viewer=(0,0)):
    view.addModel(M_true.xyzfile(), "xyz", viewer=viewer)
    view.setStyle({'sphere':{'scale': 0.5, 'opacity': 0.7}}, viewer=viewer)

    view.addModel(M_pred.xyzfile(), "xyz", viewer=viewer)
    view.setStyle({'model': -1}, {'sphere':{'scale': 0.2}}, viewer=viewer)
    for i in range(M_pred.coords.shape[0]):
        start = M_true.coords[i]
        for q, cond in enumerate(M_pred.cond_mask[i]):
            if cond:
                dx = torch.zeros_like(M_pred.coords[i])
                dx[q] = length
                end = start + dx
                view = add(view, start, end, color=color, radius=radius, obj=obj, radiusRatio=radiusRatio, fromCap=cap, toCap=cap, viewer=viewer)
                end = start - dx
                view = add(view, start, end, color=color, radius=radius, obj=obj, radiusRatio=radiusRatio, fromCap=cap, toCap=cap, viewer=viewer)
    return view



def save_view(view, fname='test.png'):
    from pathlib import Path
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    import base64
    
    net = f'<img id="img_A"><script src="https://3Dmol.org/build/3Dmol-min.js"></script><script src="https://3Dmol.org/build/3Dmol.ui-min.js"></script>' + view._make_html()
    net = net.replace('viewer_{0}.render();'.format(view.uniqueid), 'viewer_{0}.render();\nvar png = viewer_{0}.pngURI();\ndocument.getElementById("img_A").src = png;'.format(view.uniqueid))
    p = Path('temp.html')
    (f := open(p, 'w')).write(net)
    f.close()
    p_str = str(p.resolve()).replace("\\", "/")
    driver = webdriver.Firefox()
    driver.get(f'file://{p_str}')
    
    data = driver.find_element(By.ID, f'img_A').get_attribute("src")
    data = data.split('base64,')[1]
    (f := open(fname, 'wb')).write(base64.b64decode(data))
    f.close()
    p.unlink()
    driver.close()  
