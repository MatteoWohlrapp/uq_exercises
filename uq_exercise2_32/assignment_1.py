import numpy as np
import chaospy as cp
from scipy.integrate import odeint
from matplotlib.pyplot import *
import time

from typing import Union, Optional
import numpy.typing as npt
import matplotlib.pyplot as plt
# if you want you can rely also on already implemented Oscillator class
# from utils.oscillator import Oscillator

# TODO: DONE. Fix seed
np.random.seed(42)

# to perform barycentric interpolation, we'll first compute the barycentric weights
def compute_barycentric_weights(grid: npt.NDArray) -> npt.NDArray:
    size    = len(grid)
    w       = np.ones(size)

    for j in range(1, size):
        for k in range(j):
            diff = grid[k] - grid[j]

            w[k] *= diff
            w[j] *= -diff

    for j in range(size):
        w[j] = 1./w[j]

    return w


# rewrite Lagrange interpolation in the first barycentric form
def barycentric_interp(eval_point: Union[float, npt.NDArray], grid: Union[list, npt.NDArray],
 weights: Union[list, npt.NDArray], func_eval: Union[list, npt.NDArray]) -> float:
    interp_size = len(func_eval)
    L_G         = 1.
    res         = 0.

    for i in range(interp_size):
        L_G   *= (eval_point - grid[i])

    for i in range(interp_size):
        if abs(eval_point - grid[i]) < 1e-10:
            res = func_eval[i]
            L_G    = 1.0
            break
        else:
            res += (weights[i]*func_eval[i])/(eval_point - grid[i])

    res *= L_G 

    return res


# to use the odeint function, we need to transform the second order differential equation
# into a system of two linear equations
def model(init_cond: tuple[float, float], t: Union[float, npt.NDArray], args: tuple[float, float, float, float]) -> list[float]:
    x1, x2 = init_cond
    c, k, f, w = args
    f = [x2, f * np.cos(w * t) - k * x1 - c * x2]
    return f


# discretize the oscillator using the odeint function
def discretize_oscillator_odeint(model, atol: float, rtol: float, init_cond: tuple[float, float], args: tuple[float, float, float, float], 
t: npt.NDArray, t_interest: int) -> float:
    sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)
    return sol[t_interest, 0]


if __name__ == '__main__':
    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    # parameters setup as specified in the assignement
    c   = 0.5
    k   = 2.0
    f   = 0.5
    y0  = 0.5
    y1  = 0.0
    # initial conditions setup
    init_cond   = y0, y1
    # model_kwargs = {"c": c, "k": k, "f": f}  # if you want to use the Oscillator class, you can uncomment this line
    # init_cond = {"y0": y0, "y1": y1}  # if you want to use the Oscillator class, you can uncomment this line

    # time domain setup
    t_max       = 10.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t_grid          = np.array([i*dt for i in range(grid_size)])
    #t_grid = np.arange(0, t_max + dt, dt)
    t_interest  = -1

    # TODO: DONE: w is no longer deterministic ω ∼U(0.95,1.05)
    w_left      = 0.95
    w_right     = 1.05
    stat_ref    = [-0.43893703, 0.00019678]

    # TODO: DONE: create (chaospy) uniform distribution object
    w_distribution = cp.Uniform(w_left, w_right)

    # set the number of samples for Monte Carlo sampling
    no_grid_points_vec = [2, 5, 10, 20]
    # set the number of grid points for building Lagrange interpolation
    no_samples_vec = [10, 100, 1000, 10000]

    # create vectors that contain the expectations and variances (for Lagrange+MC and for only MC)
    err_exps_lagrange = np.zeros( (len(no_grid_points_vec), len(no_samples_vec)) )
    err_vars_lagrange = np.zeros( (len(no_grid_points_vec), len(no_samples_vec)) )
    err_exps_mcs = np.zeros(len(no_samples_vec))
    err_vars_mcs = np.zeros(len(no_samples_vec))

    # create vectors for storing time measurements (for Lagrange+MC and for only MC)
    lagrange_time = np.zeros( (len(no_grid_points_vec), len(no_samples_vec)) )
    mc_time = np.zeros(len(no_samples_vec))

    # compute relative error
    relative_error = lambda approx, ref: np.abs(1. - approx/ref)

    # TODO: builde interpolation-based surrogate model and comparing the stat. computed using the surrogate with a simple Monte Carlo sampling
    # iterate over vector containing different numbers of interpolation points
    for j, no_grid_points in enumerate(no_grid_points_vec):
        # TODO: a) Create the interpolant and evaluate the integral on the Lagrange interpolant using MC
        
        # TODO: DONE: a.1) generate the uniform grid and/or Chebyshev grid (i.e., experiments with one or the other),
        # Chebyshev grid 0.5(a+b)+0.5(b−a)·cos((2i−1)/(2.∗N)·π),i= 1,...,N
        grid_w = np.array([
            0.5 * (w_left + w_right) + 0.5 * (w_right - w_left) * np.cos((2 * k - 1) / (2 * no_grid_points) * np.pi)
            for k in range(1, no_grid_points + 1)
        ])
        
        # TODO: DONE: a.2) evaluate the function, and perform the interpolation
        # Evaluate the function
        f_grid = np.array([
            discretize_oscillator_odeint(model, atol, rtol, init_cond, (c, k, f, w), t_grid, t_interest) for w in grid_w
        ])
        # To perform barycentric interpolation, we'll first compute the barycentric weights
        weights = compute_barycentric_weights(grid_w)
        
        # TODO: b) Evaluate the integral directly using MC sampling
        # TODO: DONE: c) compute expectation and variance and measure runtime
        for i, no_samples in enumerate(no_samples_vec):
            samples = w_distribution.sample(no_samples, rule="random")
            
            if j == 0: # Compute MC Baseline once for each number of grid points
                ###### USE ONLY MONTE CARLO (without an interpolant) ######
                start_time = time.time()
                
                mc_evals = np.array([
                    discretize_oscillator_odeint(model, atol, rtol, init_cond, (c, k, f, w), t_grid, t_interest)
                    for w in samples
                ])
                
                mc_time[i] = time.time() - start_time

                exp_mc = np.mean(mc_evals)
                var_mc = np.var(mc_evals)

                err_exps_mcs[i] = relative_error(exp_mc, stat_ref[0])
                err_vars_mcs[i] = relative_error(var_mc, stat_ref[1])
            
            ###### USE LAGRANGE INTERPOLANTS ######
            start_time = time.time()
            
            # Get f(x) using Lagrange interpolation
            # surrogate_evals = lagrange_interpolated_evals
            surrogate_evals = np.array([
                barycentric_interp(w_val, grid_w, weights, f_grid)
                for w_val in samples
            ])
            
            lagrange_time[j, i] = time.time() - start_time
            
            exp_surrogate = np.mean(surrogate_evals)
            var_surrogate = np.var(surrogate_evals)

            err_exps_lagrange[j, i] = relative_error(exp_surrogate, stat_ref[0])
            err_vars_lagrange[j, i] = relative_error(var_surrogate, stat_ref[1])

    ###### PLOT RESULTS ######

    # Plot relative error in expectation (err_exps_lagrange vs err_exps_mcs)
    plt.figure(figsize=(10,6))
    for j, no_grid_points in enumerate(no_grid_points_vec):
        plt.plot(no_samples_vec, err_exps_lagrange[j], label=f'Lagrange, grid={no_grid_points}')
    plt.plot(no_samples_vec, err_exps_mcs, label='Monte Carlo')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Relative Error in Expectation')
    plt.title('Relative Error in Expectation vs Number of Samples')
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

    # Plot relative error in variance (err_vars_lagrange vs err_vars_mcs)
    plt.figure(figsize=(10,6))
    for j, no_grid_points in enumerate(no_grid_points_vec):
        plt.plot(no_samples_vec, err_vars_lagrange[j], label=f'Lagrange, grid={no_grid_points}')
    plt.plot(no_samples_vec, err_vars_mcs, 'k--', label='Monte Carlo')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Relative Error in Variance')
    plt.title('Relative Error in Variance vs Number of Samples')
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

    # Plot runtime comparison (lagrange_time vs mc_time)
    plt.figure(figsize=(10,6))
    for j, no_grid_points in enumerate(no_grid_points_vec):
        plt.plot(no_samples_vec, lagrange_time[j], label=f'Lagrange, grid={no_grid_points}')
    plt.plot(no_samples_vec, mc_time, 'k--', label='Monte Carlo')
    plt.xscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Seconds')
    plt.title('Runtime vs Number of Samples')
    plt.legend()
    plt.grid(True, which="both")
    plt.show()
