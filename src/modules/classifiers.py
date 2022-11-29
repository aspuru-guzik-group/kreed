import torch.nn as nn
from torch.distributions.normal import Normal

from src.kraitchman import rotated_to_principal_axes


class KraitchmanClassifier(nn.Module):

    def __init__(self, scale, stable):
        super().__init__()

        self.scale = scale
        self.stable = stable

    def forward(self, G):
        labels = G.ndata["abs_xyz"]
        G = rotated_to_principal_axes(G, stable=self.stable)

        p_clf = Normal(loc=G.ndata["xyz"].abs(), scale=self.scale)
        return p_clf.log_prob(labels)
