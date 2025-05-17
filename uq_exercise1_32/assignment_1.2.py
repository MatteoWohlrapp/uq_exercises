import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import compute_rmse

def sample_normal(
    n_samples: int, mu_target: npt.NDArray, V_target: npt.NDArray, seed: int = 42
) -> npt.NDArray:
    # TODO: generate samples from multivariate normal distribution.
    # ====================================================================
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean=mu_target,
                                      cov=V_target,
                                      size=n_samples)
    # ====================================================================
    return samples


def compute_moments(samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    # TODO: estimate mean and covariance of the samples.
    # ====================================================================
    n = len(samples)
    # Take all entries in a column, loop over the rows
    mean = np.mean(samples, axis=0)
    # Transpose the data with samples.T, because in our input a feature is a column, but np.cov expects a feature to be a row
    # Alternatively, set the rowvar to False, default is True
    # bias=False is the default settins, and means that normalization is by (n-1), as requested by exercise sheet
    covariance = np.cov(samples, bias=False, rowvar=False)
    print(f"N = {n}: mean {mean}, covariance {covariance.tolist()}")
    # ====================================================================
    return mean, covariance

def compute_rmse(covariance: npt.NDArray, n_samples: int) -> npt.NDArray:
    # RMSE is computed for each feature
    # RMSE depends on number of samples per feature
    # In a covariance matrix the entries on the diagonal are variances of each feature
    # std_dev = sqrt(variance)
    # RMSE = std_dev(feature)/sqrt(n_samples)
    std_devs = np.sqrt(np.diag(covariance))
    rmse = std_devs / np.sqrt(n_samples)
    print(f"N = {n_samples}: rmse {rmse.tolist()}")
    return std_devs / np.sqrt(n_samples)

if __name__ == "__main__":
    # ====================================================================
    # TODO: define the parameters of the simulation.
    # ====================================================================
    bivariate_mean = np.array([-0.4, 1.1], dtype=float)
    bivariate_covariance_matrix_V = np.array([[2, 0.4], [0.4, 1]], dtype=float)
    # Generate N = [10, 100, 1000, 10000] samples
    samples_10 = sample_normal(n_samples=10, mu_target=bivariate_mean, V_target=bivariate_covariance_matrix_V)
    print(f"samples_10 {samples_10}")
    samples_100 = sample_normal(n_samples=100, mu_target=bivariate_mean, V_target=bivariate_covariance_matrix_V)
    samples_1000 = sample_normal(n_samples=1000, mu_target=bivariate_mean, V_target=bivariate_covariance_matrix_V)
    samples_10000 = sample_normal(n_samples=10000, mu_target=bivariate_mean, V_target=bivariate_covariance_matrix_V)
    # # ====================================================================
    # # TODO: estimate mean, covariance, and compute the required errors.
    # # ====================================================================
    mean_10, covariance_10 = compute_moments(samples=samples_10)
    mean_100, covariance_100 = compute_moments(samples=samples_100)
    mean_1000, covariance_1000 = compute_moments(samples=samples_1000)
    mean_10000, covariance_10000 = compute_moments(samples=samples_10000)
    
    Ns = [10, 100, 1000, 1000]
    # Absolute error for first mean
    true_mean_1 = bivariate_mean[0]
    abs_mean_1_error = [abs(m[0] - true_mean_1) for m in [mean_10, mean_100, mean_1000, mean_10000]]
    # First covariance value on diagonal
    true_var_diagonal_1 = bivariate_covariance_matrix_V[0, 0]
    abs_var_diagonal_1_error = [abs(c[0, 0] - true_var_diagonal_1) for c in [covariance_10, covariance_100, covariance_1000, covariance_10000]]
    # First covariance value one off-diagonal
    true_cov_not_diagonal_1 = bivariate_covariance_matrix_V[0, 1]
    abs_cov_not_diagonal_1_error = [abs(c[0, 1] - true_cov_not_diagonal_1) for c in [covariance_10, covariance_100, covariance_1000, covariance_10000]]
    # RMSE
    rmse_of_mean_estimators_10 = compute_rmse(covariance=covariance_10, n_samples=10)
    rmse_of_mean_estimators_100 = compute_rmse(covariance=covariance_100, n_samples=100)
    rmse_of_mean_estimators_1000 = compute_rmse(covariance=covariance_1000, n_samples=1000)
    rmse_of_mean_estimators_10000 = compute_rmse(covariance=covariance_10000, n_samples=10000)
    # ====================================================================
    # TODO: plot the results on the log-log scale.
    # ====================================================================
    # Plot absolute errors
    plt.loglog(Ns, abs_mean_1_error, 'o-', label='Error of Mean 1')
    plt.loglog(Ns, abs_var_diagonal_1_error, 's--', label='Error of Var of Mean 1')
    plt.loglog(Ns, abs_cov_not_diagonal_1_error, 'x-.', label='Error of Cov Between Mean 1 and Mean 2)')
    plt.xlabel('Number of Samples (N)')
    plt.ylabel('Absolute Error')
    plt.grid(True, which="both")
    plt.legend()
    plt.show()    
    # Plot RMSE for first mean
    plt.loglog(Ns, [rmse_of_mean_estimators_10[0], rmse_of_mean_estimators_100[0], rmse_of_mean_estimators_1000[0], rmse_of_mean_estimators_10000[0]], 'o-', label='RMSE of Mean Estimator For Mean 1')
    plt.xlabel('Number of Samples (N)')
    plt.ylabel('RMSE')
    plt.grid(True, which="both")
    plt.xticks(Ns, [r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$"])
    plt.legend()
    plt.show()
    # ====================================================================
