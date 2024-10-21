import os
import numpy as np
from HamiltonianSnippets.sampler import hamiltonian_snippet
from HamiltonianSnippets.utils import eps_to_str
import pickle
import pandas as pd
from numba import jit
import scipy as sp
from copy import deepcopy


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
    mu_mean = np.repeat(mu, dim)  # np.ones((dim, 1)) * mu

    # Covariance matrix and its inverse
    covariance_matrix = f_covariance_matrix(N, beta, sigma2, dim)
    covar_reshaped = covariance_matrix.reshape(dim, dim)
    inv_covar = np.linalg.inv(covar_reshaped)
    covar_diag = np.diag(covar_reshaped)
    inv_covar_diag = 1 / np.diag(covar_reshaped)

    # Metric tensor
    metric_tensor = deepcopy(inv_covar)  # (d, d)
    np.fill_diagonal(metric_tensor, (1/dim)*np.exp(mu + 0.5*np.diag(covar_reshaped)) + np.diag(inv_covar))
    metric_tensor_diag = (1/dim)*np.exp(mu + covar_diag) + inv_covar_diag

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
        'Y': make_grid(pd.read_csv('df_pines.csv'), N),  # (dim, )
        'metric_tensor': metric_tensor,
        'metric_tensor_diag': metric_tensor_diag,
        'inv_metric_tensor_diag': 1/metric_tensor_diag,
        'metric_tensor_scheduling_func_diag': lambda gamma: (gamma/dim)*np.exp(mu + covar_diag) + inv_covar_diag,
        #'metric_tensor_scheduling_func': lambda gamma: inv_covar + (gamma/dim)*np.exp(mu + 0.5*sigma2)*np.eye(dim)
        'metric_tensor_scheduling_func': lambda gamma: inv_covar + np.diag((gamma/dim)*np.exp(mu + covar_diag)) #lambda gamma: inv_covar + (gamma/dim)*np.exp(mu + 0.5*sigma2)*np.eye(dim)  # lambda gamma: inv_covar + np.diag((gamma/dim)*np.exp(mu + covar_diag))
    }
    return parameters


def grad_neg_log_likelihood(x, parameters):
    """Computes the gradient of the negative log likelihood."""
    # New version. It might be that for those "not okay" it is enough to set the gnlls to zero
    # Recall GNLLs are used only for the leapfrog step. If for i\in[1,N] some dimension has overflown
    # then that means that that dimension will overflow for the entire trajectory. I should keep it "overflown"
    # so that I can check for it later
    gnlls = np.full(fill_value=np.inf, shape=x.shape)  # (N, d)
    ok = np.all(x < np.log(np.finfo(np.float64).max), axis=1)  # (N, ) these have all dimensions that won't overflow
    gnlls[ok] = - parameters['Y'] + np.exp(x[ok] - 2*np.log(parameters['N']))
    return gnlls


def neg_log_likelihood(x, parameters):
    """Computes the negative log likelihood. Expects x of shape (N, d).
    Notice that when x is -np.inf then the exponential goes to zero, and so really the whole result goes to np.inf.
    However, when x is np.inf then exp dominates and therefore the expression goes to -np.sum(-np.inf) or to np.inf.
    This means that to avoid RuntimeWarning and NaN values coming up, we can compute the expression only when x is
    different from np.inf."""
    nlls = np.full(fill_value=np.inf, shape=x.shape[0])  # initialize as infinity, meaning exp(-nlls) is zero
    ok = np.all(np.isfinite(x), axis=1)  # flag indicating which indices have all dimensions not `np.inf`
    ok = ok & np.all(x - 2*np.log(parameters['N']) < np.log(np.finfo(np.float64).max), axis=1)
    # notice that when x is -np.inf AND Y is 0 -np.inf * 0 throws an error. However, notice that exp(x[ok]) goes to zero
    # and although in theory x[ok] * Y = NaN, in practice we know that numerical errors like this should be assigned to
    # zero likelihood, or infinite nlls
    nlls[ok] = - np.sum(x[ok] * parameters['Y'] - np.exp(x[ok] - 2*np.log(parameters['N'])), axis=1)
    return nlls


def neg_log_prior(x, parameters):
    """Compute log prior density. Expects x of shape (N, d).
    Inv covar is (d, d) and mu_mean is (d, 1)."""
    meaned_x = x - parameters['mu_mean']  # (N, d)
    return 0.5*np.einsum('ij,ij->i', meaned_x, np.linalg.solve(parameters['covar'], meaned_x.T).T) - parameters['lognormconst_prior']


def grad_neg_log_prior(x, parameters):
    """Computes the gradient of the negative log prior."""
    return np.linalg.solve(parameters['covar'], (x - parameters['mu_mean']).T).T


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
    mass_diag = params['metric_tensor_diag']  # np.diag(params['metric_tensor'])

    # Instantiate functions
    sample_prior = generate_sample_prior_function(dim=grid_dim, mu=params['mu'], l_covar=params['l_covar'])

    n_runs = 20
    overall_seed = np.random.randint(low=0, high=10000000000)
    seeds = np.random.default_rng(overall_seed).integers(low=1, high=10000000000, size=n_runs)
    step_sizes = np.array(np.geomspace(start=0.001, stop=10.0, num=9, endpoint=True))  # np.array() used only for pylint

    # Settings
    N = 500
    T = 50
    skewness = 3
    aoo = False  # whether to act on overflow
    skipo = False  # skip overflown trajectories in epsilon computation
    verbose = False
    plot_contractivity = False
    plot_Q_criterion = False
    adapt_T = True
    adapt_epsilon = True
    adapt_mass = False
    T_max = 200
    T_min = 2
    max_contractivity = 1.5
    bottom_quantile_val = 0.05
    max_tries_find_coupling = 20
    mass_params = {
        'strategy': 'schedule',
        'matrix_type': 'full',
        'mass': params['inv_covar'],
        'schedule_func': params['metric_tensor_scheduling_func']
    }

    results = []
    for i in range(n_runs):
        print(f"Run: {i}")
        for eps_ix, eps in enumerate(step_sizes):
            epsilon_params = {
                'distribution': 'inv_gauss',
                'skewness': skewness,
                'mean': eps,
                'params_to_estimate': {'mean': lambda epsilon: epsilon},
                'to_print': 'mean',
                'on_overflow': lambda param_dict: {'skewness': max(1, param_dict['skewness'] * 0.99)},
                'param_for_T_adaptation': 'mean'
            }
            res = {'N': N, 'T': T, 'epsilon_params': {key: value for key, value in epsilon_params.items() if key not in ["params_to_estimate", "on_overflow"]}}
            out = hamiltonian_snippet(N=N, T=T, ESSrmin=0.8,
                                      sample_prior=sample_prior,
                                      compute_likelihoods_priors_gradients=lambda x: nlp_gnlp_nll_and_gnll(x, params),
                                      epsilon_params=epsilon_params,
                                      mass_params=mass_params,
                                      act_on_overflow=aoo,
                                      skip_overflown=skipo,
                                      adapt_step_size=adapt_epsilon,
                                      adapt_n_leapfrog_steps=adapt_T,
                                      plot_contractivity=plot_contractivity,
                                      plot_Q_criterion=plot_Q_criterion,
                                      T_max=T_max,
                                      T_min=T_min,
                                      max_tries_find_coupling=max_tries_find_coupling,
                                      max_contractivity=max_contractivity,
                                      bottom_quantile_val=bottom_quantile_val,
                                      verbose=verbose,
                                      seed=seeds[i])
            print(f"\t\tEps: {eps: .7f} \tLogLt: {out['logLt']: .1f} \tFinal ESS: {out['ess'][-1]: .1f}"
                  f"\tEps {epsilon_params['to_print'].capitalize()}: {out['epsilon_params_history'][-1][epsilon_params['to_print']]} "
                  f"\tFinal T: {out['T_history'][-1]}"
                  f"\tSeed {int(seeds[i])}")
            res.update({'logLt': out['logLt'], 'out': out})
            results.append(res)

    # with open(f"results/new_adaptT_cox{grid_dim}_seed{overall_seed}_N{N}_T{T}_massFalse_runs{n_runs}_from{eps_to_str(min(step_sizes))}_to{eps_to_str(max(step_sizes))}_skewness{skewness}_aoo{aoo}_skipo{skipo}_minT{T_min}.pkl", "wb") as file:
    #     pickle.dump(results, file)
