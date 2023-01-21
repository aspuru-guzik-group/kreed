import torch

from src.datamodules.qm9 import QM9Datamodule
from src.diffusion.dynamics import EquivariantDynamics
from copy import deepcopy
import numpy as np


theta = np.pi / 4
rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], 
                    [0, 0, 1]])
rot_mat = torch.tensor(rot_mat).float()

def test_reflection_model():

    data = QM9Datamodule(100, 2, zero_com=False)
    G = data.datasets['train'][0]

    dynamics = EquivariantDynamics('reflect', 16, 256, 4)
    t = torch.tensor([.2])
    model_x = dynamics(G, t)

    # reflect G
    G2 = deepcopy(G)
    G2.ndata['xyz'][:, 0] *= -1
    reflect_then_model_x = dynamics(G2, t)

    model_then_reflect_x = model_x.clone()
    model_then_reflect_x[:, 0] *= -1

    # test that f(Ref(x)) == Ref(f(x))
    assert ((model_then_reflect_x - reflect_then_model_x).abs() < 1e-5).all()

    theta = np.pi / 4
    rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0], 
                        [0, 0, 1]])
    rot_mat = torch.tensor(rot_mat).float()

    G3 = deepcopy(G)
    G3.ndata['xyz'] = G3.ndata['xyz'] @ rot_mat

    rotate_then_model_x = dynamics(G3, t)

    model_then_rotate_x = model_x @ rot_mat

    # test that f(R(x)) != R(f(x))
    assert ((model_then_rotate_x - rotate_then_model_x).abs() > 1e-5).any()

def test_rotation_model():

    data = QM9Datamodule(100, 2, zero_com=True)
    G = data.datasets['train'][0]

    dynamics = EquivariantDynamics('e3', 16, 256, 4)
    t = torch.tensor([.2])
    model_x = dynamics(G, t)

    # reflect G
    G2 = deepcopy(G)
    G2.ndata['xyz'][:, 0] *= -1
    reflect_then_model_x = dynamics(G2, t)

    model_then_reflect_x = model_x.clone()
    model_then_reflect_x[:, 0] *= -1

    # test that f(Ref(x)) == Ref(f(x))
    assert ((model_then_reflect_x - reflect_then_model_x).abs() < 1e-5).all()

    theta = np.pi / 4
    rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0], 
                        [0, 0, 1]])
    rot_mat = torch.tensor(rot_mat).float()

    G3 = deepcopy(G)
    G3.ndata['xyz'] = G3.ndata['xyz'] @ rot_mat

    rotate_then_model_x = dynamics(G3, t)

    model_then_rotate_x = model_x @ rot_mat

    # test that f(R(x)) == R(f(x))
    assert ((model_then_rotate_x - rotate_then_model_x).abs() < 1e-5).all()
