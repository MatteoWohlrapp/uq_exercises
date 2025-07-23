import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.wiener import WienerProcess

def plot_eigenpairs(
    wiener: WienerProcess, n_terms: int, t_grid: npt.NDArray[np.float64]
) -> plt.Figure:
    """Plots the first n_terms eigenvalues and eigenfunctions of the Wiener process."""
    eigenvalues, eigenfunctions = wiener.kl_eigenpairs(n_terms)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(1, n_terms + 1), eigenvalues, marker="o")
    axes[0].set_yscale("log")
    axes[0].set_title(f"First {n_terms} eigenvalues")

    # Note: I changed this to fit my impl of eigenfunctions as a list of functions
    # axes[1].plot(t_grid, eigenfunctions(t_grid))
    eigenfunction_values = []
    for t in t_grid:
        evaluations = [eigenfunctions[i](t) for i in range(len(eigenfunctions))]
        eigenfunction_values.append(evaluations)
    axes[1].plot(t_grid, eigenfunction_values)
    axes[1].set_title(f"First {n_terms} eigenfunctions")
    plt.show()
    return fig


if __name__ == "__main__":
    # TODO: set the configuration.
    T = 1.0
    n_points = 1000
    t_grid = np.linspace(0, T, n_points)
    Ms = [10, 100, 1000]
    seed = 42
    n_samples = 3
    rng = np.random.default_rng(seed)

    # TODO: generate one realization of the Wiener process using the
    # standard definition.
    wiener = WienerProcess(mu=0, T=T, n_points=n_points)
    wiener_standard = wiener.generate(n_samples, rng)
    print(f"wiener_standard: {wiener_standard}")

    # TODO: generate approximations of the Wiener process using the KL expansion.
    realizations = np.zeros((3, n_points))
    for i, M in enumerate(Ms):
        rng = np.random.default_rng(seed)
        approx = wiener.approximate_kl(1, M, rng)
        realizations[i, :] = approx

    # TODO: plot the approximation results.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, M in enumerate(Ms):
        axes[i].plot(t_grid, realizations[i, :])
        axes[i].set_title(f"Approximation with {M} terms")
    plt.show()

    # Plot the standard Wiener process
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(t_grid, wiener_standard[0, :])
    ax.set_title("Standard Wiener process")
    plt.show()

    # TODO: visualize first eigenvalues and eigenfunctions.
    plot_eigenpairs(wiener, 10, t_grid)