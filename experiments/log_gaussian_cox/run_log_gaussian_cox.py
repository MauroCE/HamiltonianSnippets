import numpy as np
from HamiltonianSnippets.sampler import hamiltonian_snippet
from HamiltonianSnippets.utils import eps_to_str
import pickle
import pandas as pd
from numba import jit


def generate_sample_prior_function(dim, mu, l_covar):
    """Samples from the prior distribution."""
    def sample_prior_function(n_particles, random_number_generator):
        white_noise = random_number_generator.normal(loc=0.0, scale=1.0, size=(n_particles, dim))
        return white_noise.dot(l_covar.T) + mu
    return sample_prior_function


@jit(nopython=True)
def covariance_function(i, j, i_prime, j_prime, beta, sigma2, dim):
    """Compute the covariance between two points (i, j) and (i', j') on a 2D grid based on the given parameters.

    Parameters:
    - i, j: Indices of the first point on the 2D grid.
    - i_prime, j_prime: Indices of the second point on the 2D grid.
    - beta: Scaling factor for the covariance function.
    - sigma2: Variance parameter of the Gaussian process.
    - dim: Total number of grid points (dim = N^2).

    Returns:
    - res: Covariance value between the two points.
    """
    return sigma2 * np.exp(-np.sqrt(((i - i_prime) ** 2) + ((j - j_prime) ** 2)) / (np.sqrt(dim) * beta))


@jit(nopython=True)
def f_covariance_matrix(N, beta, sigma2, dim):
    """Compute the covariance matrix of the Gaussian process on a 2D grid.

    Parameters:
    - N: Number of grid points along one dimension.
    - beta: Scaling factor for the covariance function.
    - sigma2: Variance parameter of the Gaussian process.
    - dim: Total number of grid points (dim = N^2).

    Returns:
    - covariance_matrix: 4D tensor representing the covariance between grid points.
    """
    covariance_matrix = np.zeros((N, N, N, N))
    for i in range(N):
        for i_prime in range(N):
            for j in range(N):
                for j_prime in range(N):
                    covariance_matrix[i, j, i_prime, j_prime] = covariance_function(i, j, i_prime, j_prime, beta, sigma2, dim)
    return covariance_matrix


def make_grid(data, N):
    """Generates a grid from the given data by counting the number of points in each cell of an NxN grid.

    Parameters:
        - data: DataFrame with columns ['data_x', 'data_y'] representing spatial coordinates.
        - N: Number of grid points along one dimension (total grid will be N x N).
    Returns:
        - data_counts: Array of shape (N^2,), the count of points in each grid cell.
    """
    grid = np.linspace(start=0, stop=1, num=N+1)
    dim = N**2
    data_counts = np.zeros(dim)
    data_x = data['data_x']
    data_y = data['data_y']
    for i in range(N):
        for j in range(N):
            logical_x = (data_x > grid[i]) & (data_x < grid[i + 1])
            logical_y = (data_y > grid[j]) & (data_y < grid[j + 1])
            data_counts[i * N + j] = sum(logical_y & logical_x)
    return data_counts


def create_log_cox_parameters(N):
    """Create and initialize the parameters for the Log Gaussian Cox process.

    Parameters:
    - N: Number of grid points along one dimension.

    Returns:
    - parameters: Dictionary containing all necessary parameters for the Log Cox model.
    """
    beta = 1. / 33.
    sigma2 = 1.91
    mu = np.log(126.) - sigma2 / 2.
    dim = N ** 2
    mu_mean = np.ones((dim, 1)) * mu

    # Covariance matrix and its inverse
    covariance_matrix = f_covariance_matrix(N, beta, sigma2, dim)
    covar_reshaped = covariance_matrix.reshape(dim, dim)
    inv_covar = np.linalg.inv(covar_reshaped)

    # Metric tensor
    metric_tensor = inv_covar
    np.fill_diagonal(metric_tensor, (1/dim)*np.exp(mu + 0.5*np.diag(covar_reshaped)) + np.diag(inv_covar))

    # Define parameters dictionary
    parameters = {
        'N': N,
        'beta': beta,
        'sigma2': sigma2,
        'mu': mu,
        'dim': dim,
        'mu_mean': mu_mean,
        'covar': covar_reshaped,
        'inv_covar': inv_covar,
        'lognormconst_prior': -0.5 * np.linalg.slogdet(covar_reshaped)[1] - 0.5 * dim * np.log(2 * np.pi),
        'l_covar': np.linalg.cholesky(covar_reshaped),
        'Y': make_grid(pd.read_csv('df_pines.csv'), N)[:, np.newaxis],
        'metric_tensor': metric_tensor,
        'inv_metric_tensor': np.linalg.inv(metric_tensor)
    }
    return parameters


def grad_neg_log_likelihood(x, parameters):
    """Computes the gradient of the negative log likelihood."""
    return - parameters['Y'].ravel() + np.exp(x - 2*np.log(parameters['N']))  # (1./parameters['dim'])*np.exp(x)  # (N, d)


def neg_log_likelihood(x, parameters):
    """Computes the negative log likelihood. Expects x of shape (N, d)."""
    return - np.sum(x * parameters['Y'].ravel() - np.exp(x - 2*np.log(parameters['N'])), axis=1)


def neg_log_prior(x, parameters):
    """Compute log prior density. Expects x of shape (N, d).
    Inv covar is (d, d) and mu_mean is (d, 1)."""
    meaned_x = x - parameters['mu_mean'].ravel()  # (N, d)
    return 0.5*np.einsum('ij,ij->i', meaned_x @ parameters['inv_covar'], meaned_x) - parameters['lognormconst_prior']


def grad_neg_log_prior(x, parameters):
    """Computes the gradient of the negative log prior."""
    return parameters['inv_covar'].dot(x.transpose() - parameters['mu_mean']).transpose()  # (N, d)


def nlp_gnlp_nll_and_gnll(x, parameters):
    """Computes negative log prior, its gradient, negative log likelihood and its gradient."""
    nlp = neg_log_prior(x, parameters)  # (N, ) negative log prior
    gnlp = grad_neg_log_prior(x, parameters)  # (N, d) gradient negative log prior
    nll = neg_log_likelihood(x, parameters)  # (N, ) negative log likelihood
    gnll = grad_neg_log_likelihood(x, parameters)  # (N, d) gradient negative log likelihood
    return nlp, gnlp, nll, gnll


if __name__ == "__main__":
    # Load data and parameters
    grid_dim = 400
    params = create_log_cox_parameters(N=int(grid_dim**0.5))
    # M = Sigma^{-1} + (1/grid_dim)*exp(mu + Sigma_diag)
    # Sigma_diag = np.diag(params['covar'])
    # mass_diag = (1 / Sigma_diag) + (1/grid_dim)*np.exp(params['mu'] + Sigma_diag)
    mass_diag = np.diag(params['metric_tensor'])
    # mass_diag = #np.ones(grid_dim)  # 1 / np.diag(params['covar'])

    # Instantiate functions
    sample_prior = generate_sample_prior_function(dim=grid_dim, mu=params['mu'], l_covar=params['l_covar'])

    # Settings
    N = 500
    T = 30
    skewness = 1
    epsilon_params = {
        'distribution': 'inv_gauss',
        'skewness': skewness,
        'mean': 0.01,
        'params_to_estimate': {'mean': lambda epsilon: epsilon},
        'to_print': 'mean'
    }

    out = hamiltonian_snippet(N=N, T=T, mass_diag=mass_diag, ESSrmin=0.8, sample_prior=sample_prior,
                              compute_likelihoods_priors_gradients=lambda x: nlp_gnlp_nll_and_gnll(x, params),
                              epsilon_params=epsilon_params,
                              adapt_mass=False, verbose=True, seed=1234)


