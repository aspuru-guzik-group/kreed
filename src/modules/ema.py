import copy

from torch import nn


class EMA(nn.Module):

    def __init__(self, model, beta=0.9999):
        super().__init__()

        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        self.beta = beta

    def update(self, model):
        for p, p_ema in zip(model.parameters(), self.ema_model.parameters()):
            p_ema.data.lerp_(p, weight=(1.0 - self.beta))
