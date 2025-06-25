import numpy as np
import chaospy as cp
from scipy.integrate import odeint
from matplotlib.pyplot import *
import time
import matplotlib.pyplot as plt  

from typing import Union, Optional
import numpy.typing as npt
# if you want you can rely also on already implemented Oscillator class
# from utils.oscillator import Oscillator

# TODO: DONE. Fix seed
np.random.seed(42)

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
    ### deterministic setup ###

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

    ### stochastic setup ####
    # w is no longer deterministic
    w_left      = 0.95
    w_right     = 1.05
    # TODO: create uniform distribution object
    distr_w = cp.Uniform(w_left, w_right)

    # the truncation order of the polynomial chaos expansion approximation
    N = [1, 2, 3, 4, 5, 6]
    # the quadrature degree of the scheme used to computed the expansion coefficients
    K = [1, 2, 3, 4, 5, 6]

    assert(len(N)==len(K))
    
    # vector to save the statistics
    exp_m = np.zeros(len(N))
    var_m = np.zeros(len(N))

    exp_cp = np.zeros(len(N))
    var_cp = np.zeros(len(N))

    # extra: timing & reference for plots
    time_m = np.zeros(len(N))
    time_cp_arr = np.zeros(len(N))

    # high-accuracy Monte-Carlo reference
    n_mc = 100000
    mc_w = distr_w.sample(n_mc)
    mc_vals = np.array([discretize_oscillator_odeint(model, atol, rtol, init_cond,
                                                     (c, k, f, float(w)), t_grid, t_interest) for w in mc_w])
    exp_ref = mc_vals.mean()
    var_ref = mc_vals.var(ddof=0)

    # perform polynomial chaos approximation + the pseudo-spectral
    for h in range(len(N)):

        # TODO: create N[h] orthogonal polynomials using chaospy
        poly = cp.orth_ttr(N[h]-1, distr_w)          # length N[h]

        # TODO: create K[h] quadrature nodes using chaospy
        nodes, weights  = cp.generate_quadrature(K[h], distr_w, rule="gaussian")
        nodes = nodes.flatten()                      # (Q,)

        # model evaluations on quadrature nodes
        tic = time.time()
        f_vals = np.array([discretize_oscillator_odeint(model, atol, rtol, init_cond,
                                                        (c, k, f, float(w)), t_grid, t_interest) for w in nodes])

        # TODO: perform polynomial chaos approximation + the pseudo-spectral approach manually
        norms = cp.E(poly**2, distr_w)
        coeffs = np.zeros(N[h])
        for i in range(N[h]):
            coeffs[i] = np.sum(f_vals * poly[i](nodes) * weights) / norms[i]
        exp_m[h] = coeffs[0]
        var_m[h] = np.sum(coeffs[1:]**2 * norms[1:])
        time_m[h] = time.time() - tic

        # TODO: perform polynomial chaos approximation + the pseudo-spectral approach using chaospy
        tic = time.time()
        gpc = cp.fit_quadrature(poly, nodes, weights, f_vals)
        exp_cp[h] = cp.E(gpc, distr_w)
        var_cp[h] = cp.Var(gpc, distr_w)
        time_cp_arr[h] = time.time() - tic
        
    print('MEAN')
    print("K | N | Manual \t\t\t| ChaosPy")
    for h in range(len(N)):
        print(K[h], '|', N[h], '|', "{a:1.12f}".format(a=exp_m[h]), '\t|', "{a:1.12f}".format(a=exp_cp[h]))

    print('VARIANCE')
    print("K | N | Manual \t\t| ChaosPy")
    for h in range(len(N)):
        print(K[h], '|', N[h], '|', "{a:1.12f}".format(a=var_m[h]), '\t|', "{a:1.12f}".format(a=var_cp[h]))

    ###### PLOT RESULTS ######
    rel_err_exp_m = np.abs((exp_m - exp_ref)/exp_ref)
    rel_err_exp_cp = np.abs((exp_cp - exp_ref)/exp_ref)
    rel_err_var_m = np.abs((var_m - var_ref)/var_ref)
    rel_err_var_cp = np.abs((var_cp - var_ref)/var_ref)

    # Plot relative error in expectation
    plt.figure(figsize=(10,6))
    plt.plot(N, rel_err_exp_m, 'o-', label='Manual')
    plt.plot(N, rel_err_exp_cp, 's--', label='ChaosPy')
    plt.yscale('log')
    plt.xlabel('Truncation Order N')
    plt.ylabel('Relative Error in Expectation')
    plt.title('Relative Error in Expectation vs Truncation Order')
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig('relative_error_expectation.png')

    # Plot relative error in variance
    plt.figure(figsize=(10,6))
    plt.plot(N, rel_err_var_m, 'o-', label='Manual')
    plt.plot(N, rel_err_var_cp, 's--', label='ChaosPy')
    plt.yscale('log')
    plt.xlabel('Truncation Order N')
    plt.ylabel('Relative Error in Variance')
    plt.title('Relative Error in Variance vs Truncation Order')
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig('relative_error_variance.png')

    # Plot runtime comparison
    plt.figure(figsize=(10,6))
    plt.plot(N, time_m, 'o-', label='Manual')
    plt.plot(N, time_cp_arr, 's--', label='ChaosPy')
    plt.xlabel('Truncation Order N')
    plt.ylabel('Seconds')
    plt.title('Runtime vs Truncation Order')
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig('runtime_comparison.png')