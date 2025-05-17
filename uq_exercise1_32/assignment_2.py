from functools import partial
from typing import Callable

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import monte_carlo

Function = Callable[[float], float]


"""
a, b = -1, 2
distr = cp.Uniform(a, b)
# Sample from the defined distribution.
sample_size = 1000
samples = distr.sample(size=sample_size)
# To ensure reproducibility, pass the 'seed' parameter.
# You can also provide additional parameters to control
# the generation.
samples = distr.sample(size=sample_size, seed=42)
"""


def f(x: float) -> float:
    # TODO: define the target function.
    # ====================================================================
    return np.sin(x)
    # ====================================================================


def analytical_integral(a: float, b: float) -> float:
    # TODO: compute the analytical integral of f on [a, b].
    # ====================================================================
    return np.cos(a) - np.cos(b)
    # ====================================================================


def transform(samples: npt.NDArray, a: float, b: float) -> npt.NDArray:
    # TODO: implement the transformation of U from [0, 1] to [a, b].
    # ====================================================================
    transformed_samples = a + (b - a) * samples
    # ====================================================================
    return transformed_samples


def integrate_mc(
    f: Function,
    a: float,
    b: float,
    n_samples: int,
    with_transform: bool = False,
    seed: int = 42,
) -> tuple[float, float]:
    # TODO: compute the integral with the Monta Carlo method.
    # Depending on 'with_transform', use the uniform distribution on [a, b]
    # directly or transform the uniform distribution on [0, 1] to [a, b].
    # Return the integral estimate and the corresponding RMSE.
    # ====================================================================
    if with_transform:
        distr = cp.Uniform(0, 1)
    else: 
        distr = cp.Uniform(a, b)

    samples = distr.sample(size=n_samples, seed=seed)

    if with_transform:
        samples = transform(samples, a, b)

    integral = (b - a) * np.mean(f(samples))
    rmse = (b - a) * np.std(f(samples)) / np.sqrt(n_samples)

    # ====================================================================
    return integral, rmse


if __name__ == "__main__":

    # Assignment 2.1
    # TODO: define the parameters of the simulation.
    # ====================================================================
    N = [10, 100, 1000, 10000]
    # ====================================================================

    # TODO: compute the integral and the errors.
    # ====================================================================
    integrals = []
    rmses = []
    for n in N:
        integral, rmse = integrate_mc(f, 0, 1, n)
        integrals.append(integral)
        rmses.append(rmse)

    integral_analytical = analytical_integral(0, 1)
    exact_errors = np.abs(np.array(integrals) - integral_analytical)
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    plt.loglog(N, exact_errors, label="Exact Error")
    plt.loglog(N, rmses, label="RMSE")
    plt.legend()
    plt.savefig("report/21_rmse_exact_error.png")
    # ====================================================================

    # Assignment 2.2
    integrals_wo_transform = []
    rmses_wo_transform = []
    for n in N:
        integral, rmse = integrate_mc(f, 2, 4, n, with_transform=False)
        integrals_wo_transform.append(integral)
        rmses_wo_transform.append(rmse)

    integral_analytical = analytical_integral(2, 4)
    exact_errors_wo_transform = np.abs(np.array(integrals_wo_transform) - integral_analytical)

    plt.clf()
    plt.loglog(N, exact_errors_wo_transform, label="Exact Error")
    plt.loglog(N, rmses_wo_transform, label="RMSE")
    plt.legend()
    plt.savefig("report/22_rmse_exact_error_wo_transform.png")

    integrals_with_transform = []
    rmses_with_transform = []
    for n in N:
        integral, rmse = integrate_mc(f, 2, 4, n, with_transform=True)
        integrals_with_transform.append(integral)
        rmses_with_transform.append(rmse)

    exact_errors_with_transform = np.abs(np.array(integrals_with_transform) - integral_analytical)

    plt.clf()
    plt.loglog(N, exact_errors_with_transform, label="Exact Error")
    plt.loglog(N, rmses_with_transform, label="RMSE")
    plt.legend()
    plt.savefig("report/22_rmse_exact_error_with_transform.png")