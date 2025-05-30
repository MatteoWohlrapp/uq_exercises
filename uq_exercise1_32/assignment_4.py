from collections import defaultdict

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.oscillator import Oscillator


def load_reference(filename: str) -> tuple[float, float]:
    # Load reference values for the mean and variance.
    with open(filename, "r") as file:
        mean, var = map(float, file.read().splitlines())
    return mean, var


def simulate(
    t_grid: npt.NDArray,
    omega_distr: cp.Distribution,
    n_samples: int,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
    rule="random",
    seed=42,
) -> npt.NDArray:
    # Simulate the oscillator with the given parameters and return
    # generated solutions.
    c, k, f = model_kwargs["c"], model_kwargs["k"], model_kwargs["f"]
    y0, y1 = init_cond["y0"], init_cond["y1"]

    # The rule argument is either "random" or "halton" and will make sure we sample the omega values correctly
    assert rule in ["random", "halton"]
    omegas = omega_distr.sample(n_samples, rule)

    solutions = []
    for omega in omegas:
        oscillator = Oscillator(c, k, f, omega)
        solution = oscillator.discretize("odeint", y0, y1, t_grid)
        solutions.append(solution)

    return np.array(solutions)


def compute_errors(
    samples: npt.NDArray, mean_ref: float, var_ref: float
) -> tuple[float, float]:
    # Compute the relative errors of the mean and variance
    # estimates.

    # First, compute mean and variance of y(10) (last value in the time grid) for each method across the M runs
    # Samples is of shape (N, len(t_grid)), want the mean/var of y(10) across all N outcomes that are based on different omega
    y10s = samples[:, -1]
    mean = np.mean(y10s)
    var = np.var(y10s)
    rel_mean_error = np.abs(1 - mean / mean_ref)
    rel_var_error = np.abs(1 - var / var_ref)
    return rel_mean_error, rel_var_error


if __name__ == "__main__":
    # Define the parameters of the simulations.
    c = 0.5
    k = 2.0
    f = 0.5
    y0 = 0.5
    y1 = 0.0
    t_delta = 0.01
    t_start = 0
    t_stop = 10
    t_grid = np.arange(t_start, t_stop + t_delta, t_delta)
    print(f"shape of t_grid: {t_grid.shape}")
    print(f"t_grid: {t_grid}")
    Ns = [10, 100, 1000, 10000]

    # Deterministic case
    omega = 1.0
    # Stochastic case
    omega_distr = cp.Uniform(0.95, 1.05)

    # Run the deterministic case
    oscillator = Oscillator(c, k, f, omega)
    deterministic_solutions = oscillator.discretize("odeint", y0, y1, t_grid)
    print(f"Deterministic solutions: {deterministic_solutions}")

    # Work on stochastic cases; omega is sampled from a distribution and we have M=4 simulations with N samples each
    # Therefore, the solutions will be of shape (M, N, len(t_grid))
    mc_solutions = []
    quasi_mc_solutions = []
    for n in Ns:
        # Run the stochastic case with standard MC
        mc_solution = simulate(t_grid, omega_distr, n, {"c": c, "k": k, "f": f}, {"y0": y0, "y1": y1}, rule="random")
        mc_solutions.append(mc_solution)
        # Run the stochastic case with Quasi-MC based on Halton sequences
        quasi_mc_solution = simulate(t_grid, omega_distr, n, {"c": c, "k": k, "f": f}, {"y0": y0, "y1": y1}, rule="halton")
        quasi_mc_solutions.append(quasi_mc_solution)


    # Now: Compute the RELATIVE errors, comparing to reference values; want to plot rel error over the # samples N for each method
    # Keep getting confused about the bigger picture and the role of M and N here -> clarification:
    #   We want to estimate y(10) while omega is uncertain. Therefore, we sample it N times from a distribution and see how the output changes.
    #   We can compute mean and variance of a specific outcome, e.g. final outcome at y(10), across the N samples.
    #   We do this M times where we vary N (e.g. N=10, 100, 1000, 10000) to see if we get a better estimate.
    #   Whether we the estimate for that Mth run across N samples is better, we can verify via the relative error to some reference values.

    # Get reference values for y(10)
    mean_ref, var_ref = load_reference("data/oscillator_ref.txt")

    # # Compute mean and variance of y(10) (last value in the time grid) for each method across the M runs
    # mc_means = [] # shape (M, )
    # mc_vars = [] # shape (M, )
    # for sim_run in mc_solutions:
    #     # sim_run is of shape (N, len(t_grid))
    #     # want the mean/var of y(10) across all N outcomes that are based on different omega
    #     y10s = sim_run[:, -1]
    #     mean = np.mean(y10s)
    #     var = np.var(y10s)
    #     mc_means.append(mean)
    #     mc_vars.append(var)
    # quasi_mc_means = [] # shape (M, )
    # quasi_mc_vars = [] # shape (M, )
    # for sim_run in quasi_mc_solutions:
    #     # sim_run is of shape (N, len(t_grid))
    #     y10s = sim_run[:, -1]
    #     mean = np.mean(y10s)
    #     var = np.var(y10s)
    #     quasi_mc_means.append(mean)
    #     quasi_mc_vars.append(var)

    # Compare to ref
    errors_mc = [compute_errors(mc_solution, mean_ref, var_ref) for mc_solution in mc_solutions] # shape should be (M, 2)
    errors_quasi_mc = [compute_errors(quasi_mc_solution, mean_ref, var_ref) for quasi_mc_solution in quasi_mc_solutions] # shape should be (M, 2)
    errors_mc = np.array(errors_mc)
    errors_quasi_mc = np.array(errors_quasi_mc)

    # Plot the results on the log-log scale.
    figure, axes = plt.subplots()
    axes.loglog(Ns, errors_mc[:, 0], label="MC")
    axes.loglog(Ns, errors_quasi_mc[:, 0], label="Quasi-MC")
    axes.set_title("Relative mean error")
    axes.legend()
    plt.show()

    figure, axes = plt.subplots()
    axes.loglog(Ns, errors_mc[:, 1], label="MC")
    axes.loglog(Ns, errors_quasi_mc[:, 1], label="Quasi-MC")
    axes.set_title("Relative variance error")
    axes.legend()
    plt.show()

    # Plot 10 sampled trajectories from standard MC. Also adding deterministic solution as reference
    figure, axes = plt.subplots()
    axes.plot(t_grid, deterministic_solutions, label="Deterministic")
    for i in range(10):
        axes.plot(t_grid, mc_solutions[0][i], label=f"MC {i}") # mc_solutions is of shape (M, N, len(t_grid))
    axes.legend()
    plt.show()