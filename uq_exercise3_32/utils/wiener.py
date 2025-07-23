from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class WienerProcess:
    mu: float
    T: float | None = None
    n_points: float | None = None
    t_grid: npt.NDArray | None = None

    def __post_init__(self):
        if self.T is None and self.n_points is None:
            self.T = self.t_grid[-1]
            self.n_points = len(self.t_grid)
        if self.t_grid is None:
            self.t_grid = np.linspace(0, self.T, self.n_points)

    def generate(self, n_samples: int, rng: np.random.Generator):
        # TODO: generate n_samples realizations of the Wiener process
        # using the standard definition.

        # Note that n_samples counts the realizations whereas n_points indicates how many random variables we look at
        # In the post init, we have already set the t_grid, which is an evenly spaced grid we can use to sample the Wiener process
        delta_t = self.t_grid[1] - self.t_grid[0]
        print(f"delta_t: {delta_t}")
        result = np.zeros((n_samples, self.n_points))
        for sample in range(n_samples):
            for i, t in enumerate(self.t_grid):
                if i == 0:
                    result[sample, i] = self.mu
                else:
                    increment = rng.normal(0, delta_t)
                    result[sample, i] = result[sample, i-1] + increment
        return result

    def approximate_kl(self, n_samples: int, M: int, rng: np.random.Generator):
        # TODO: generate n_samples realizations of the Wiener process
        # using the Karhunen-Loève expansion with M terms.

        # RECAP KL-expansion:
        # 1) Mercer's theorem: Can express the Wiener process via Eigenpairs and an uncorrelated random variable (have to truncate sum at some point for approx.)
        # 2) To compute Eigenpairs, need to solve continuos Eigenvalue problem (thanks to 2nd kind Fredholm integral + Nyström method)
        # However: here, explicit formulas are given to compute Eigenpairs
        # Here, M determines how many terms we use to approximate the Wiener process; there is also a decision to be made in the quadrature usually but because we
        #  have an explicit way to compute the Eigenpairs, that approximation is not needed here

        # Compute the M first Eigenpairs
        eigenvalues, eigenfunctions = self.kl_eigenpairs(M)
        # Compute zetas
        zetas = rng.normal(0, 1, (n_samples, M))

        result = np.zeros((n_samples, self.n_points))
        # Compute n_samples realizations of the Wiener process based on the M Eigenpairs
        for sample in range(n_samples):
            for i, t in enumerate(self.t_grid):
                W_t = 0
                for m in range(M):
                    W_t += np.sqrt(eigenvalues[m]) * eigenfunctions[m](t) * zetas[sample, m]
                # W_t += self.mu * t
                result[sample, i] = W_t
        return result


    def kl_eigenvalues(self, M: int):
        lambdas = np.zeros(M)
        for m in range(M):
            lambda_m = 1 / ((m - 0.5) ** 2 * np.pi ** 2)
            lambdas[m] = lambda_m
        return lambdas

    def kl_eigenfunctions(self, M: int):
        # TODO: compute the first M eigenfunctions of the Wiener process.
        # It might be more conveniet to return a callable function that
        # returns evaluations of the first M eigenfunctions for the provided
        # time points.

        eigenfunctions = [
            lambda t, m=m: np.sqrt(2) * np.sin(np.pi * t * (m - 0.5)) for m in range(1, M + 1)
        ]

        # TODO: write eigenfunctions the way they want it for visulization
        return eigenfunctions
        # return lambda _: np.zeros(M)

    def kl_eigenpairs(self, M: int):
        eigenvalues = self.kl_eigenvalues(M)
        eigenfunctions = self.kl_eigenfunctions(M)
        return eigenvalues, eigenfunctions
