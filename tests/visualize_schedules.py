import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.diffusion.schedules import LearnedNoiseSchedule, FixedNoiseSchedule


def visualize_schedules():
    T = 1000

    schedules = {
        "polynomial_1": FixedNoiseSchedule("polynomial_1", timesteps=T, precision=1e-5),
        "polynomial_2": FixedNoiseSchedule("polynomial_2", timesteps=T, precision=1e-5),
        "cosine": FixedNoiseSchedule("cosine", timesteps=T, precision=0.008),
        "learned_init": LearnedNoiseSchedule(),
        "learned_trained": LearnedNoiseSchedule(),
    }

    gt = schedules["polynomial_2"]
    nn = schedules["learned_trained"]
    optim = torch.optim.SGD(nn.parameters(), lr=1e-2)
    for _ in range(2000):
        nn.zero_grad()
        loss = F.mse_loss(nn.sweep(T)[1], gt.sweep(T)[1])
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
    plt.show()


if __name__ == "__main__":
    visualize_schedules()
