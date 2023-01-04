import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import trange

from src.modules.schedules import LearnedNoiseSchedule, FixedNoiseSchedule


def plot_schedules():
    T = 1000

    schedules = {
        "polynomial_1": FixedNoiseSchedule("polynomial_1", timesteps=T, precision=1e-5),
        "polynomial_2": FixedNoiseSchedule("polynomial_2", timesteps=T, precision=1e-5),
        "cosine": FixedNoiseSchedule("cosine", timesteps=T, precision=0.008),
        "learned_initial": LearnedNoiseSchedule(d_hidden=10),
        "learned_trained": LearnedNoiseSchedule(d_hidden=10),
    }

    model = schedules["learned_trained"]
    optim = torch.optim.SGD(model.parameters(), lr=1e-2)

    labels = schedules["polynomial_2"].sweep(T)[1]
    for _ in trange(1000, desc="Training Schedule"):
        model.zero_grad()
        loss = F.mse_loss(model.sweep(T)[1], labels)
        loss.backward()
        optim.step()

    for label, schedule in schedules.items():
        t, gammas = schedule.sweep(T + 1)
        t = t.detach().cpu().numpy()
        alphas_cumprod = torch.sigmoid(-gammas).detach().cpu().numpy()
        plt.plot(t, alphas_cumprod, label=label)

    plt.xlabel(r"$t$")
    plt.ylabel(r"$\bar{\alpha}_t$")

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_schedules()
