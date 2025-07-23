import time
from functools import partial
from typing import Callable

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.oscillator import Oscillator
from utils.wiener import WienerProcess


def generate_f_samples(
    mu: float,
    t_grid: npt.NDArray,
    n_samples: int,
    M: int | None,
    rng: np.random.Generator,
) -> list[Callable[[float], float]]:
    """Generates samples of the Wiener process."""

    # TODO: generate realizations of the Wiener process for f(t).
    # If M is None, we generate samples using the standard definition.
    # If M is not None, we generate samples using the KL expansion with M terms.
    # The samples are returned as a list of callable functions that
    # evaluate the Wiener process at a given time point.
    

    # Important: f has to be callable because the oscillator model expects that

    if M is None:
        # For the standard definition, we can reuse the code we already implemented for 3.1
        # t_grid is enough for the WienerProcess class to figure out number of points and T
        wiener = WienerProcess(mu=mu, t_grid=t_grid)
        samples = wiener.generate(n_samples, rng) # shape (n_samples, n_points)
    
    else:
        # For the KL expansion, use the provided terms on the worksheet
        samples = np.zeros((n_samples, len(t_grid)))
        # Draw the zetas already, they are independent of t
        zetas = rng.normal(0, 1, (n_samples, M))
        for i in range(n_samples):
            for j, t in enumerate(t_grid):
                W_t = mu
                for m in range(1, M + 1):
                    phi_m = np.sqrt(2 / t_grid[-1]) * np.sin(((m + 0.5) * np.pi * t) / t_grid[-1])
                    lambda_m = (t_grid[-1] ** 2) / ((m + 0.5) ** 2 * np.pi ** 2)
                    # Have to do -1 to index zetas correctly because m starts at 1 but our generated list is obviously 0-indexed
                    W_t += np.sqrt(lambda_m) * phi_m * zetas[i, m - 1]
                # Saving W_t for that specific t (indexed by j) for that sample (indexed by i)
                samples[i, j] = W_t

    # Now we need to convert the samples to a list of callable functions
    # that evaluate the Wiener process at a given time point
    # Have to map any t to the corresponding index in t_grid
    return [lambda t, i=i: samples[i, np.argmin(np.abs(t_grid - t))] for i in range(n_samples)]


def simulate(
    t_grid: npt.NDArray,
    f_samples: list[Callable[[float], float]],
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> npt.NDArray:
    """Simulates the oscillator model for each sample of f(t)."""

    # TODO: simulate the oscillator model for each sample of f(t) and
    # return the trajectories as 2D array.

    solutions = []

    for f_sample in f_samples:
        oscillator = Oscillator(c=model_kwargs["c"], k=model_kwargs["k"], f=f_sample, omega=model_kwargs["omega"])
        solution = oscillator.discretize(method="euler", y0=init_cond["y0"], y1=init_cond["y1"], t_grid=t_grid)
        solutions.append(solution)
    
    return np.array(solutions)
    # return np.zeros(len(f_samples), len(t_grid))


def compute_metrics(solutions: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Computes the mean and standard deviation of the solutions."""

    # TODO: compute the metrics.
    return np.zeros(solutions.shape[1]), np.zeros(solutions.shape[1])


def plot_solutions(
    t_grid: npt.NDArray, sampler_solutions: dict[str, npt.NDArray]
) -> plt.Figure:
    """Plots the oscillator trajectories for each sample of f."""
    n_plots = len(sampler_solutions)
    fig, axes = plt.subplots(
        1, n_plots, figsize=(6 * n_plots, 4), sharex=True, sharey=True
    )
    for ax, (name, solutions) in zip(axes, sampler_solutions.items()):
        mean, std = compute_metrics(solutions)
        ax.plot(t_grid, solutions.T, alpha=0.01, c="b")
        ax.plot(t_grid, mean, c="r", label="mean")
        ax.fill_between(
            t_grid, mean - std, mean + std, color="red", alpha=0.5, label="std"
        )

        # Add legend for samples manually.
        handles, _ = ax.get_legend_handles_labels()
        line = lines.Line2D([0], [0], color="b", label="Monte Carlo samples")
        handles.append(line)
        ax.legend(handles=handles)

        ax.set_title(name)
    return fig


if __name__ == "__main__":
    # TODO: set parameters of the model.
    f_mean = 0.5
    model_kwargs = {"c": 0.5, "k": 2.0, "omega": 1.0}
    init_cond = {"y0": 0.5, "y1": 0.0}

    # TODO: set the time domain.
    T_max = 10
    dt = 0.01
    t_grid = np.arange(0, T_max + dt, dt)

    # TODO: set the number of Monte-Carlo samples and KL terms.
    N = 1000
    Ms = [5, 10, 100]
    seed = None
    rng = np.random.default_rng(seed)

    ###########################################################################

    # TODO: generate samples of the Wiener process for f using the stadard
    # generation and the KL expansion for different M.

    # Generate samples with standard definition of Wiener process
    # TODO: again seeing the magnitude difference between standard Wiener process definition and KL expansion, bug somewhere
    samples_standard = generate_f_samples(f_mean, t_grid, N, None, rng)
    samples_kl_5 = generate_f_samples(f_mean, t_grid, N, 5, rng)
    samples_kl_10 = generate_f_samples(f_mean, t_grid, N, 10, rng)
    samples_kl_100 = generate_f_samples(f_mean, t_grid, N, 100, rng)

    # TODO: simulate the oscillator model for each sample of f and record the
    # mean and standard deviation of the solutions at T_max.
    
    # Simulate the oscillator model for the Wiener process standard samples
    solutions_standard = simulate(t_grid, samples_standard, model_kwargs, init_cond)
    print(f"solutions_standard shape: {solutions_standard.shape}")
    print(f"solutions_standard: {solutions_standard[:, -1]}")

    # Simulate based on KL expansion
    solutions_kl_5 = simulate(t_grid, samples_kl_5, model_kwargs, init_cond)
    solutions_kl_10 = simulate(t_grid, samples_kl_10, model_kwargs, init_cond)
    solutions_kl_100 = simulate(t_grid, samples_kl_100, model_kwargs, init_cond)

    print(f"solutions_kl_5: {solutions_kl_5[:, -1]}")
    print(f"solutions_kl_10: {solutions_kl_10[:, -1]}")
    print(f"solutions_kl_100: {solutions_kl_100[:, -1]}")

    # TODO: optionally, plot the solutions for each sample of f.

    # Compute mean and variance at the last timestep for each way of generating f
    mean_standard = np.mean(solutions_standard[:, -1])
    std_standard = np.std(solutions_standard[:, -1])
    mean_kl_5 = np.mean(solutions_kl_5[:, -1])
    std_kl_5 = np.std(solutions_kl_5[:, -1])
    mean_kl_10 = np.mean(solutions_kl_10[:, -1])
    std_kl_10 = np.std(solutions_kl_10[:, -1])
    mean_kl_100 = np.mean(solutions_kl_100[:, -1])
    std_kl_100 = np.std(solutions_kl_100[:, -1])

    print(f"mean_standard: {mean_standard}")
    print(f"std_standard: {std_standard}")
    print(f"mean_kl_5: {mean_kl_5}")
    print(f"std_kl_5: {std_kl_5}")
    print(f"mean_kl_10: {mean_kl_10}")
    print(f"std_kl_10: {std_kl_10}")
    print(f"mean_kl_100: {mean_kl_100}")
    print(f"std_kl_100: {std_kl_100}")

    # Create a plt that I can visualize, showing the trajectories
    fig, ax = plt.subplots()
    ax.plot(t_grid, solutions_standard.T, label="Standard")
    ax.plot(t_grid, solutions_kl_5.T, label="KL 5")
    ax.plot(t_grid, solutions_kl_10.T, label="KL 10")
    ax.plot(t_grid, solutions_kl_100.T, label="KL 100")
    ax.legend()
    plt.show()

