import time

import chaospy as cp
import numpy as np
from scipy.integrate import odeint
from matplotlib.pyplot import figure, bar, xticks, title, tight_layout, show, savefig, subplot, subplots, legend

from utils.sobol import monte_carlo_sobol, pseudo_spectral_sobol


def discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest):
	sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)

	return sol[t_interest, 0]

def get_distribution(
    c_lims: tuple[float, float],
    k_lims: tuple[float, float],
    f_lims: tuple[float, float],
    y0_lims: tuple[float, float],
    y1_lims: tuple[float, float],
) -> cp.Distribution:
    """Creates the joint distribution over the stochastic parameters."""

    # create uniform distribution object
    distr_c     = cp.Uniform(c_lims[0], c_lims[1])
    distr_k     = cp.Uniform(k_lims[0], k_lims[1])
    distr_f     = cp.Uniform(f_lims[0], f_lims[1])
    distr_y0    = cp.Uniform(y0_lims[0], y0_lims[1])
    distr_y1    = cp.Uniform(y1_lims[0], y1_lims[1])

    # create the multivariate distribution
    return cp.J(distr_c, distr_k, distr_f, distr_y0, distr_y1)


def run_method(method, **kwargs):
    """Runs the specified method and prints the results.

    The results include the first and total order Sobol' indices as well as
    the elapsed time to run the method."""

    # TODO: run the method and print the results.
    if method == 'mc':
        n_samples = kwargs['n_samples']
        distribution = kwargs['distribution']
        t_grid = kwargs['t_grid']
        fixed_args = kwargs['fixed_args']
        first_order_indices, total_order_indices = monte_carlo_sobol(n_samples, distribution, t_grid, fixed_args)
    elif method == 'ps':
        pce_degree = kwargs['pce_degree']
        quadrature_degree = kwargs['quadrature_degree']
        distribution = kwargs['distribution']
        t_grid = kwargs['t_grid']
        fixed_args = kwargs['fixed_args']
        sparse = kwargs['sparse']
        first_order_indices, total_order_indices = pseudo_spectral_sobol(pce_degree, quadrature_degree, distribution, t_grid, fixed_args, sparse)
    else:
        raise ValueError(f"Invalid method: {method}")
    
    return first_order_indices, total_order_indices

if __name__ == "__main__":
    # TODO: set the stochastic parameters.
    c_lims = [0.08, 0.12]
    k_lims = [0.03, 0.04]
    f_lims = [0.08, 0.12]
    y0_lims = [0.45, 0.55]
    y1_lims = [-0.05, 0.05]

    # TODO: set the determinisic parameters.
    fixed_args = {"omega": 1, "atol": 1e-10, "rtol": 1e-10, "t_interest": 10}

    # TODO: set the parameters of the methods.
    quadrature_degree = 4
    pce_degree = 4
    n_samples = (quadrature_degree + 1) ** 5

    # TODO: set the time domain
    T_max = 10
    dt = 0.01
    t_grid = np.arange(0, T_max + dt, dt)

    ###########################################################################

    # TODO: define the distribution over the stochastic parameters.
    distr_5D = get_distribution(c_lims, k_lims, f_lims, y0_lims, y1_lims)
    
    # TODO: run the pseudo-spectral method on full grid.
    full_grid_first_order_indices, full_grid_total_order_indices = run_method('ps', pce_degree=pce_degree, quadrature_degree=quadrature_degree, distribution=distr_5D, t_grid=t_grid, fixed_args=fixed_args, sparse=False)

    # TODO: run the pseudo-spectral method on sparse grid.
    sparse_grid_first_order_indices, sparse_grid_total_order_indices = run_method('ps', pce_degree=pce_degree, quadrature_degree=quadrature_degree, distribution=distr_5D, t_grid=t_grid, fixed_args=fixed_args, sparse=True)
    
    # TODO: run the Monte Carlo method.
    mc_first_order_indices, mc_total_order_indices = run_method('mc', n_samples=n_samples, distribution=distr_5D, t_grid=t_grid, fixed_args=fixed_args)


    ###########################################################################
    Sobol_indices_x = np.arange(5)  # [0, 1, 2, 3, 4] for positioning
    labels = ['c', 'k', 'f', r"$y_0$", r"$y_1$"]
    
    # Bar width and positions for grouped bars - made thinner with more space
    bar_width = 0.25
    spacing = 0.05  # Additional spacing between groups
    x_first = Sobol_indices_x - bar_width/2 - spacing/2
    x_total = Sobol_indices_x + bar_width/2 + spacing/2
    
    # Use a better colormap
    import matplotlib.pyplot as plt
    colors = plt.cm.tab10(np.linspace(0, 1, 5))  # Vibrant, colorful palette
    
    # Create grouped bar chart for full grid
    fig1, ax1 = subplots(figsize=(10, 6))
    
    bars1 = ax1.bar(x_first, full_grid_first_order_indices, bar_width, 
                    label='First Order', color=colors, alpha=0.9, edgecolor='black', linewidth=0.8,
                    hatch='///')  # Diagonal stripes for first order
    bars2 = ax1.bar(x_total, full_grid_total_order_indices, bar_width, 
                    label='Total Order', color=colors, alpha=1.0, edgecolor='black', linewidth=0.8)
    
    ax1.set_xlabel('Parameters', fontsize=12)
    ax1.set_ylabel('Sensitivity Index', fontsize=12)
    ax1.set_title("Sobol' Indices Comparison (Full Grid)", fontsize=14, fontweight='bold')
    ax1.set_xticks(Sobol_indices_x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(max(full_grid_first_order_indices), max(full_grid_total_order_indices)) * 1.1)
    
    tight_layout()
    savefig(f"ps_full_{pce_degree}_{quadrature_degree}.png", dpi=300, bbox_inches='tight')

    # Create grouped bar chart for sparse grid
    fig2, ax2 = subplots(figsize=(10, 6))
    
    bars3 = ax2.bar(x_first, sparse_grid_first_order_indices, bar_width, 
                    label='First Order', color=colors, alpha=0.9, edgecolor='black', linewidth=0.8,
                    hatch='///')  # Diagonal stripes for first order
    bars4 = ax2.bar(x_total, sparse_grid_total_order_indices, bar_width, 
                    label='Total Order', color=colors, alpha=1.0, edgecolor='black', linewidth=0.8)
    
    ax2.set_xlabel('Parameters', fontsize=12)
    ax2.set_ylabel('Sensitivity Index', fontsize=12)
    ax2.set_title("Sobol' Indices Comparison (Sparse Grid)", fontsize=14, fontweight='bold')
    ax2.set_xticks(Sobol_indices_x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max(max(sparse_grid_first_order_indices), max(sparse_grid_total_order_indices)) * 1.1)
    
    tight_layout()
    savefig(f"ps_sparse_{pce_degree}_{quadrature_degree}.png", dpi=300, bbox_inches='tight')
    
    # Create grouped bar chart for Monte Carlo
    fig3, ax3 = subplots(figsize=(10, 6))
    
    bars5 = ax3.bar(x_first, mc_first_order_indices, bar_width, 
                    label='First Order', color=colors, alpha=0.9, edgecolor='black', linewidth=0.8,
                    hatch='///')  # Diagonal stripes for first order
    bars6 = ax3.bar(x_total, mc_total_order_indices, bar_width, 
                    label='Total Order', color=colors, alpha=1.0, edgecolor='black', linewidth=0.8)
    
    ax3.set_xlabel('Parameters', fontsize=12)
    ax3.set_ylabel('Sensitivity Index', fontsize=12)
    ax3.set_title("Sobol' Indices Comparison (Monte Carlo)", fontsize=14, fontweight='bold')
    ax3.set_xticks(Sobol_indices_x)
    ax3.set_xticklabels(labels, fontsize=11)
    ax3.legend(fontsize=11, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim(0, max(max(mc_first_order_indices), max(mc_total_order_indices)) * 1.1)
    
    tight_layout()
    savefig(f"mc_{n_samples}.png", dpi=300, bbox_inches='tight')
    