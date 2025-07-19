from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def exp_cov_fn(x: npt.NDArray, y: npt.NDArray, scale: float) -> npt.NDArray:
    """Computes the exponential covariance function between two sets of points."""
    # TODO: compute the exponential covariance function.
    # x[:, None, :] <- insert dummy dimension (N, 1, D)
    # y[None, :, :] <- insert dummy dimension (1, M, D)
    # x[:, None, :] - y[None, :, :] <- pairwise diff (N, M, D)
    # np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1) <- norm on last dim (N, M)
    distances = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    result = np.exp(- distances / scale)
    return result


def squared_exp_cov_fn(x: npt.NDArray, y: npt.NDArray, scale: npt.NDArray):
    """Computes the squared exponential covariance function between two sets of points."""
    # TODO: compute the squared exponential covariance function.
    distances_squared = np.sum((x[:, None] - y[None, :])**2, axis=-1)
    result = np.exp(- distances_squared / (2 * scale**2))
    return result


def get_xy_mesh(
    x_lims: tuple[float, float],
    y_lims: tuple[float, float],
    x_mesh_size: int,
    y_mesh_size: int,
) -> npt.NDArray:
    """Creates a 2D mesh grid for the given limits and mesh sizes."""
    x_step = (x_lims[1] - x_lims[0]) / x_mesh_size
    y_step = (y_lims[1] - y_lims[0]) / y_mesh_size
    x_grid = np.arange(x_lims[0] + x_step / 2, x_lims[1], x_step)
    y_grid = np.arange(y_lims[0] + y_step / 2, y_lims[1], y_step)
    mesh = np.stack(np.meshgrid(x_grid, y_grid), axis=-1)
    return mesh


def sample(mesh, mean_fn, cov_fn, n_samples, rng, reg_scale=1e-7):
    """Samples from a Gaussian process defined by the mean and covariance functions."""
    # TODO: sample a Gaussian field suing the Cholesky decomposition.
    print(f"mesh.shape {mesh.shape}") # (N, N, D)
    x = mesh.reshape(-1, mesh.shape[-1]) # (automatic, original) = (N**2, D)
    mean_x = mean_fn(x)
    print(f"mean_x.shape {mean_x.shape}")
    cov_x = cov_fn(x, x)
    print(f"cov_x.shape {cov_x.shape}")
    # add some small numbers to the diagonal entries of the cov marix
    cov_x += reg_scale * np.eye(cov_x.shape[0])
    # Cholesky decomposition, L is lower triagular matrix
    L = np.linalg.cholesky(cov_x)
    print(f"L.shape {L.shape}")
    # Obtain samples from Gas Gi = ˆ m+ LΨ, where Ψ ∼N(0N 2 ,IN 2 ), and IN 2 is the identity matrix of size N2 ×N2
    G = rng.standard_normal(size=(cov_x.shape[0], n_samples))
    print(f"G.shape {G.shape}")
    # @ for matrix multiplication
    samples = mean_x[:, None] + L @ G
    N = mesh.shape[0]
    samples = samples.T.reshape(n_samples, N, N)
    print(f"samples.shape {samples.shape}")
    return samples


def plot_samples(samples, x_lims, y_lims):
    """Plots the samples from the Gaussian process."""
    n_plots = len(samples)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    for ax, sample in zip(axes, samples):
        ax.imshow(sample, cmap="coolwarm", origin="lower", extent=(*x_lims, *y_lims))
    return fig


if __name__ == "__main__":
    # TODO: set the condiguration.
    N = 100
    x_lims, y_lims = (0.0, 1.0), (0.0, 1.0) # the domain [0,1]^2
    x_mesh_size, y_mesh_size = N, N # N x N grid
    
    # scale is responsible for variance/spread of curve in covariance function - possible values e.g. 5, 0.5, 1.0, 0.1
    # the higher the scale the more flat the kernel function
    scale = 1
    
    # mean function m(x) = 0.1, gives the mean vector m* = R**(N**2) 
    mean = lambda x: np.full(x.shape[0], 0.1)
    seed = 42
    
    n_samples = 3 # we need three from each field
    rng = np.random.default_rng(seed)

    # TODO: create a 2D mesh.
    mesh = get_xy_mesh(x_lims, y_lims, x_mesh_size, y_mesh_size)

    # TODO: sample from the Gaussian process with different kernels.
    
    # sample(mesh, mean_fn, cov_fn, n_samples, rng, reg_scale=1e-7)
    exp_cov_field_samples = sample(mesh=mesh, mean_fn=mean,
                                   cov_fn=lambda x, y: exp_cov_fn(x, y, scale),
                                   n_samples=n_samples, rng=rng)
    square_exp_cov_field_samples = sample(mesh=mesh, mean_fn=mean,
                                          cov_fn=lambda x, y: squared_exp_cov_fn(x, y, scale),
                                          n_samples=n_samples, rng=rng)

    # TODO: plot the samples.
    
    # plot_samples(samples, x_lims, y_lims)
    exp_cov_field_plot = plot_samples(samples=exp_cov_field_samples, x_lims=x_lims, y_lims=y_lims)
    exp_cov_field_plot.suptitle("exp_cov_field_plot")
    exp_cov_field_plot.savefig("exp_cov_field_plot.png")
    square_exp_cov_field_plot = plot_samples(samples=square_exp_cov_field_samples, x_lims=x_lims, y_lims=y_lims)
    square_exp_cov_field_plot.suptitle("square_exp_cov_field_plot")
    square_exp_cov_field_plot.savefig("square_exp_cov_field_plot.png")