import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np

from utils.sampling import control_variates, importance_sampling, monte_carlo


def f(x: float) -> float:
    return np.exp(x)


def analytical_integral() -> float:
    # Compute the analytical integral of f on [0, 1].
    return np.exp(1) - 1


def run_monte_carlo(Ns: list[int], seed: int = 42):
    # Run the Monte Carlo method and return the absolute error
    # of the estimation.
    abs_errors = []
    for n in Ns:
        samples = np.random.default_rng(seed).uniform(0, 1, n)
        approx = 0
        for sample in samples:
            approx += f(sample)
        approx /= n
        abs_errors.append(np.abs(approx - analytical_integral()))
    return abs_errors

def phi_1(x: float) -> float:
    return x

def phi_2(x: float) -> float:
    return 1 + x

def phi_3(x: float) -> float:
    return 1 + x + (x**2 / 2)


def run_control_variates(
    Ns: list[int], seed: int = 42
):
    # Run the control variate method for and return the absolute
    #  errors of the resulting estimations.

    # Need the expected values of the phi functions
    # phi_1_expected is the expected value of x; since x is uniform on [0, 1], we have E[x] = 0.5
    phi_1_expected = 0.5
    # phi_2_expected is the expected value of 1 + x, so we have to compute E[1 + x] = 1 + E[x] = 1 + 0.5 = 1.5
    phi_2_expected = 1.5
    # phi_3_expected is the expected value of 1 + x + (x^2 / 2) so E[1 + x + (x^2 / 2)] = 1 + E[x] + 0.5 * E[x^2] = 1 + 0.5 + .5 * (1/3) = 1.6666666666666667
    phi_3_expected = 1.6666666666666667

    abs_errors_1 = []
    abs_errors_2 = []
    abs_errors_3 = []
    
    # For the control variate method, the estimator changes!
    for n in Ns:
        samples = np.random.default_rng(seed).uniform(0, 1, n)
        fs = [f(sample) for sample in samples]
        phi_1s = [phi_1(sample) for sample in samples]
        phi_2s = [phi_2(sample) for sample in samples]
        phi_3s = [phi_3(sample) for sample in samples]

        f_bar = np.mean(fs)
        phi_1_bar = np.mean(phi_1s)
        phi_2_bar = np.mean(phi_2s)
        phi_3_bar = np.mean(phi_3s)

        # Need the optimal alpha values, see Tutorial 4, Task 3: alpha* = (pearsoncorr(f, phi) * sigma(f)) / sigma(phi)
        # Therefore, need to estimate the Pearson correlation and the standard deviations
        # Pearson correlation is the covariance of f and phi divided by the product of the standard deviations of f and phi
        # 1. Compute the std of f and phi
        sigma_f = np.std(fs)
        sigma_phi_1 = np.std(phi_1s)
        sigma_phi_2 = np.std(phi_2s)
        sigma_phi_3 = np.std(phi_3s)
        # 2. Compute the covariance of f and phi
        covariance_f_phi_1 = np.cov(fs, phi_1s)[0][1]
        covariance_f_phi_2 = np.cov(fs, phi_2s)[0][1]
        covariance_f_phi_3 = np.cov(fs, phi_3s)[0][1]
        # 3. Compute the Pearson correlation
        pearson_correlation_f_phi_1 = covariance_f_phi_1 / (sigma_f * sigma_phi_1)
        pearson_correlation_f_phi_2 = covariance_f_phi_2 / (sigma_f * sigma_phi_2)
        pearson_correlation_f_phi_3 = covariance_f_phi_3 / (sigma_f * sigma_phi_3)
        # 4. Compute the optimal alpha values
        alpha_1 = (pearson_correlation_f_phi_1 * sigma_f) / sigma_phi_1
        alpha_2 = (pearson_correlation_f_phi_2 * sigma_f) / sigma_phi_2
        alpha_3 = (pearson_correlation_f_phi_3 * sigma_f) / sigma_phi_3

        estimator_1 = f_bar + alpha_1 * (phi_1_expected - phi_1_bar)
        estimator_2 = f_bar + alpha_2 * (phi_2_expected - phi_2_bar)
        estimator_3 = f_bar + alpha_3 * (phi_3_expected - phi_3_bar)

        abs_errors_1.append(np.abs(estimator_1 - analytical_integral()))
        abs_errors_2.append(np.abs(estimator_2 - analytical_integral()))
        abs_errors_3.append(np.abs(estimator_3 - analytical_integral()))

    return abs_errors_1, abs_errors_2, abs_errors_3


def run_importance_sampling(
    Ns: list[int], seed: int = 42
):
    # Run the importance sampling method and return the absolute
    # errors of the resulting estimations.
    
    # For IS, we define another distribution q(x), here given as Beta distributions (will have 2 parameters)
    alpha_1, beta_1 = (5, 1)
    alpha_2, beta_2 = (0.5, 0.5)

    abs_errors_1 = []
    abs_errors_2 = []

    # Estimator is still based on sampling f but then weighting by importance weights, computed as p(x) / q(x)
    for n in Ns:
        samples_1 = cp.Beta(alpha_1, beta_1).sample(n)
        samples_2 = cp.Beta(alpha_2, beta_2).sample(n)

        # Compute the estimators; each estimator is the mean of the samples evaluated in f and "importance weighted"
        estimator_1 = np.mean([f(sample) * cp.Uniform(0, 1).pdf(sample) / cp.Beta(alpha_1, beta_1).pdf(sample) for sample in samples_1])
        estimator_2 = np.mean([f(sample) * cp.Uniform(0, 1).pdf(sample) / cp.Beta(alpha_2, beta_2).pdf(sample) for sample in samples_2])

        # Gather the absolute errors
        abs_errors_1.append(np.abs(estimator_1 - analytical_integral()))
        abs_errors_2.append(np.abs(estimator_2 - analytical_integral()))

    return abs_errors_1, abs_errors_2



if __name__ == "__main__":
    # Define the parameters of the simulation.
    Ns = [10, 100, 1000, 10000]

    # Run all the methods
    monte_carlo_abs_error = run_monte_carlo(Ns)
    control_variates_abs_error_1, control_variates_abs_error_2, control_variates_abs_error_3 = run_control_variates(Ns)
    importance_sampling_abs_error_1, importance_sampling_abs_error_2 = run_importance_sampling(Ns)

    print(f"Monte Carlo absolute error: {np.mean(monte_carlo_abs_error)}")
    print(f"Control Variates 1 absolute error: {np.mean(control_variates_abs_error_1)}")
    print(f"Control Variates 2 absolute error: {np.mean(control_variates_abs_error_2)}")
    print(f"Control Variates 3 absolute error: {np.mean(control_variates_abs_error_3)}")
    print(f"Importance Sampling 1 absolute error: {np.mean(importance_sampling_abs_error_1)}")
    print(f"Importance Sampling 2 absolute error: {np.mean(importance_sampling_abs_error_2)}")

    # Plot the results on the log-log scale.
    fig_mcs, ax_mcs = plt.subplots()
    ax_mcs.semilogx(np.array(Ns).reshape(-1, 1), monte_carlo_abs_error, label="Standard MCS")
    ax_mcs.semilogx(np.array(Ns).reshape(-1, 1), control_variates_abs_error_1, label="Control Variates 1")
    ax_mcs.semilogx(np.array(Ns).reshape(-1, 1), control_variates_abs_error_2, label="Control Variates 2")
    ax_mcs.semilogx(np.array(Ns).reshape(-1, 1), control_variates_abs_error_3, label="Control Variates 3")
    ax_mcs.semilogx(np.array(Ns).reshape(-1, 1), importance_sampling_abs_error_1, label="Importance Sampling 1")
    ax_mcs.semilogx(np.array(Ns).reshape(-1, 1), importance_sampling_abs_error_2, label="Importance Sampling 2")
    ax_mcs.set(title='Comparison of MCS techniques')
    plt.legend()
    plt.show()

    # Make another plot showing the f(x) and the two beta distributions
    fig_beta, ax_beta = plt.subplots()
    ax_beta.plot(np.linspace(0, 1, 100), [f(x) for x in np.linspace(0, 1, 100)], label="f(x)")
    ax_beta.plot(np.linspace(0, 1, 100), [cp.Beta(5, 1).pdf(x) for x in np.linspace(0, 1, 100)], label="Beta 1")
    ax_beta.plot(np.linspace(0, 1, 100), [cp.Beta(0.5, 0.5).pdf(x) for x in np.linspace(0, 1, 100)], label="Beta 2")
    ax_beta.set(title='Comparison of f(x) and the two beta distributions')
    plt.legend()
    plt.show()
