from typing import Callable

import chaospy as cp
import numpy as np
import numpy.typing as npt

Function = Callable[[npt.NDArray], npt.NDArray]


def compute_rmse(values: npt.NDArray) -> npt.NDArray:
    return np.std(values, ddof=1, axis=1) / np.sqrt(values.shape[1])


def monte_carlo(
    p: cp.Distribution,
    n_samples: int,
    f: Function,
    transform: Function | None = None,
    rule: str = "random",
    seed: float = 42,
) -> tuple[npt.NDArray, float]:
    # Implement the Monte Carlo method.
    # Return the mean approximation and the corresponding RMSE. Make sure
    # the function works for both 1-dimensional and n-dimensional
    # distributions (see include_axis_dim parameter of
    # cp.Distribution.sample).

    samples = p.sample(n_samples, rule=rule, seed=seed, include_axis_dim=True)
    approx = 0
    values = []
    for sample in samples:
        f_output = f(sample)
        values.append(f_output)
    approx = np.sum(values) / n_samples
    values = np.array(values)

    rmse = compute_rmse(values)

    return approx, rmse


def control_variates(
    p: cp.Distribution,
    n_samples: int,
    f: Function,
    phi: Function,
    control_mean: float,
    seed: float = 42,
) -> npt.NDArray:
    # Implement the control variates method that returns the mean
    # approximation. Make sure the function works for both 1-dimensional
    # and n-dimensional distributions.
    
    samples = p.sample(n_samples, seed=seed, include_axis_dim=True)
    fs = [f(sample) for sample in samples]
    phis = [phi(sample) for sample in samples]

    f_bar = np.mean(fs)
    phi_bar = np.mean(phis)

    # Need the optimal alpha values, see Tutorial 4, Task 3: alpha* = (pearsoncorr(f, phi) * sigma(f)) / sigma(phi)
    # Therefore, need to estimate the Pearson correlation and the standard deviations
    # Pearson correlation is the covariance of f and phi divided by the product of the standard deviations of f and phi
    # 1. Compute the std of f and phi
    sigma_f = np.std(fs)
    sigma_phi = np.std(phis)
    # 2. Compute the covariance of f and phi
    covariance_f_phi = np.cov(fs, phis)[0][1]
    # 3. Compute the Pearson correlation
    pearson_correlation_f_phi = covariance_f_phi / (sigma_f * sigma_phi)
    # 4. Compute the optimal alpha value
    alpha = (pearson_correlation_f_phi * sigma_f) / sigma_phi

    estimator = f_bar + alpha * (control_mean - phi_bar)

    return estimator


def importance_sampling(
    p: cp.Distribution,
    q: cp.Distribution,
    n_samples: int,
    f: Function,
    seed: float = 42,
) -> npt.NDArray:
    # Implement the importance sampling that returns the mean
    # approximation. Make sure the function works for both 1-dimensional
    # and n-dimensional distributions.
    samples = q.sample(n_samples, seed=seed, include_axis_dim=True)

    # Compute the estimators; each estimator is the mean of the samples evaluated in f and "importance weighted"
    estimator = np.mean([f(sample) * p.pdf(sample) / q.pdf(sample) for sample in samples])
    return estimator