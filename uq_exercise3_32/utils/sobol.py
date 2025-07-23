import chaospy as cp
import numpy as np
import numpy.typing as npt

from .oscillator import Oscillator


def _evaluate_oscillator(
    samples: npt.NDArray, t_grid: npt.NDArray, fixed_args: dict[str, float]
) -> npt.NDArray:
    """Evaluates the oscillator model for given samples."""

    sol_odeint = np.zeros(len(samples.T))
    dt = t_grid[1] - t_grid[0]

    for j, n in enumerate(samples.T):
        oscillator = Oscillator(c=n[0], k=n[1], f=n[2], omega=fixed_args['omega'])
        sol_odeint[j] = oscillator.discretize(method="odeint", y0=n[3], y1=n[4], t_grid=t_grid, atol=fixed_args['atol'], rtol=fixed_args['rtol'])[int(fixed_args['t_interest'] / dt)]

    return sol_odeint

def monte_carlo_sobol(
    n_samples: int,
    distribution: cp.Distribution,
    t_grid: npt.NDArray[np.float64],
    fixed_args: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    First-order and total Sobol’ indices by the Jansen (1999) / Saltelli
    (2010) estimator with the A, A(i)_B radial sampling plan.

    Returns
    -------
    first_order : (d,) array
    total_order : (d,) array
    """

    d = len(distribution)

    AB = distribution.sample(2 * n_samples, rule="Sobol")    # if we dont do this the samples are the same for A and B
    A  = AB[:, :n_samples]
    B  = AB[:, n_samples:]

    YA = _evaluate_oscillator(A, t_grid, fixed_args)      # (N,)
    YB = _evaluate_oscillator(B, t_grid, fixed_args)

    VarY = np.var(np.concatenate([YA, YB]), ddof=1)
    if VarY == 0.0:
        raise RuntimeError("The model output is (numerically) constant – "
                           "Sobol’ indices are undefined.")

    first_order = np.empty(d)
    total_order = np.empty(d)

    A_Bi = A.copy()

    for i in range(d):
        # build the hybrid matrix A(i)_B
        A_Bi[i, :] = B[i, :]
        Y_A_Bi = _evaluate_oscillator(A_Bi, t_grid, fixed_args)

        # Jansen formulas 
        # first-order
        first_order[i] = (
            np.mean(YB * (Y_A_Bi - YA))
        ) / VarY


        # total order
        total_order[i] = (
            0.5 * np.mean((YA - Y_A_Bi) ** 2)
        ) / VarY

        A_Bi[i, :] = A[i, :]

    return np.clip(first_order, 0.0, 1.0), np.clip(total_order, 0.0, 1.0)


def pseudo_spectral_sobol(
    pce_degree: int,
    quadrature_degree: int,
    distribution: cp.Distribution,
    t_grid: npt.NDArray[np.float64],
    fixed_args: dict[str, float],
    sparse=True,
) -> tuple[float, float]:
    """Computes the Sobol' indices using a pseudo-spectral method."""
    
    P = cp.generate_expansion(pce_degree, distribution, normed=True)
    nodes, weights = cp.generate_quadrature(quadrature_degree, distribution, rule='G', sparse=sparse)

    sol_odeint = _evaluate_oscillator(nodes, t_grid, fixed_args)

    sol_gpc_sparse_approx, gpc_coeffs_sparse = cp.fit_quadrature(P, nodes, weights, sol_odeint, retall=True)

    # compute first order and total Sobol' indices
    first_order_Sobol_ind_sparse   = cp.Sens_m(sol_gpc_sparse_approx, distribution)
    total_Sobol_ind_sparse         = cp.Sens_t(sol_gpc_sparse_approx, distribution)

    return first_order_Sobol_ind_sparse, total_Sobol_ind_sparse

