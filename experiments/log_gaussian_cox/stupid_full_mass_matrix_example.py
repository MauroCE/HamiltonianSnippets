"""
The aim of this script is to be self-contained. I will run Hamiltonian Snippets with a full metric tensor rather than
a diagonal matrix. Maybe that is what is making the difference.
"""
import pandas as pd
from numba import jit
import time
import numpy as np
from typing import Optional
from numpy.typing import NDArray
from scipy.stats import invgauss
from scipy.optimize import brentq
from scipy.special import logsumexp
from typing import Tuple


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


def sample_epsilons(eps_params: dict, N: int, rng: np.random.Generator) -> NDArray:
    """Samples epsilons according to the correct distribution."""
    assert 'distribution' in eps_params, "Epsilon parameters must contain a key 'distribution'."
    match eps_params['distribution']:
        case 'inv_gauss':
            assert eps_params['skewness'] > 0, "Skewness must be strictly positive for an inverse gaussian."
            assert eps_params['mean'] > 0, "Mean must be strictly positive for an inverse gaussian."
            lambda_param = 9*eps_params['mean'] / eps_params['skewness']**2  # since skewness = 3*sqrt(mu/lambda)
            return invgauss.rvs(mu=eps_params['mean']/lambda_param, loc=0, scale=lambda_param, size=N, random_state=rng)


def leapfrog(x, v, T, epsilons, gamma_curr, inv_mass_curr, compute_likelihoods_priors_gradients):
    """Leapfrog integration with full mass matrix"""
    N, d = x.shape
    if len(epsilons.shape) == 1:
        epsilons = epsilons[:, None]

    # Initialize snippets, negative log priors and negative log likelihoods
    xnk = np.full((N, T+1, d), np.nan)
    vnk = np.full((N, T+1, d), np.nan)
    nlps = np.full((N, T+1), np.nan)  # negative log priors
    nlls = np.full((N, T+1), np.nan)  # negative log likelihoods
    xnk[:, 0] = x  # seed positions
    vnk[:, 0] = v  # seed velocities

    # First momentum half-step
    nlps[:, 0], gnlps, nlls[:, 0], gnlls = compute_likelihoods_priors_gradients(x)
    v = v - 0.5*epsilons*(gnlps + gamma_curr*gnlls)  # (N, d)

    # T - 1 position and velocity full steps
    for k in range(T-1):

        # Full position step
        x = x + epsilons*(v.dot(inv_mass_curr))

        # Full momentum step
        nlps[:, k+1], gnlps, nlls[:, k+1], gnlls = compute_likelihoods_priors_gradients(x)
        v = v - epsilons*(gnlps + gamma_curr*gnlls)

        # Store positions and velocities
        vnk[:, k+1] = v
        xnk[:, k+1] = x

    # Final position half-step
    x = x + epsilons*(v.dot(inv_mass_curr))

    # Final momentum half-step
    nlps[:, -1], gnlps, nlls[:, -1], gnlls = compute_likelihoods_priors_gradients(x)
    v = v - 0.5*epsilons*(gnlps + gamma_curr*gnlls)

    # Store final position and velocity
    xnk[:, -1] = x
    vnk[:, -1] = v

    return xnk, vnk, nlps, nlls


def ess_from_log_weights(logw: np.ndarray) -> float:
    """ESS (Effective sample size) computed from log-weights, this function has been copied from Nicholas Chopin
    `particles` package.

    Parameters
    ----------
    :param logw: Log weights of shape `(N, )`
    :type logw: np.ndarray
    :return: ESS of weights `w=np.exp(logw)` i.e. the quantity `np.sum(w**2) / np.sum(w)**2`
    :rtype: float
    """
    w = np.exp(logw - logw.max())
    return (w.sum()) ** 2 / np.sum(w ** 2)


def next_annealing_param(gamma: float, ESSrmin: float, llk) -> float:
    """Find next annealing exponent by solving `ESS(exp(logw)) = alpha * N`.

    Parameters
    ----------
    :param gamma: Current tempering parameter, typically denoted $\\gamma_{n-1}$
    :type gamma: float
    :param ESSrmin: Target ESS proportion with respect to `N`, e.g. `0.8` means we target `0.8N` as our ESS
    :type ESSrmin: float
    :param llk: Log likelihoods, has shape `(N, )`
    :type llk: np.ndarray
    :return: New tempering parameter
    :rtype: float
    """
    N = llk.shape[0]

    def f(e):
        ess = ess_from_log_weights(e * llk) if e > 0.0 else N  # avoid 0 x inf issue when e==0
        return ess - ESSrmin * N
    if f(1. - gamma) < 0.:
        return gamma + brentq(f, 0.0, 1.0 - gamma)
    else:
        return 1.0


def eps_to_str(epsilon):
    """Simply replace . with 'dot'"""
    return str(float(epsilon)).replace(".", "dot")


def compute_weights(
        vnk: NDArray,
        nlps: NDArray,
        nlls: NDArray,
        inv_mass_next: NDArray,
        inv_mass_curr: NDArray,
        gamma_next: float,
        gamma_curr: float,
        overflow_mask: NDArray,
        ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Computes unfolded and folded weights.

    Parameters
    ----------
    :param vnk: Velocities constructed with $\\psi_{n-1}$, has shape `(N, T+1, d)` where `d` is the velocity dimension
    :type vnk: numpy.ndarray
    :param nlps: Negative log prior density on the snippets, has shape `(N, T+1)`
    :type nlps: numpy.ndarray
    :param nlls: Negative log likelihood on the snippets, has shape `(N, T+1)`
    :type nlls: numpy.ndarray
    :param inv_mass_next: Diagonal of the inverse mass matrix at time `n`
    :type inv_mass_next: numpy.ndarray
    :param inv_mass_curr: Diagonal of the inverse mass matrix at time `n-1`
    :type inv_mass_curr: numpy.ndarray
    :param gamma_next: Tempering parameter $\\gamma_n$ for the next distribution, will be used in the numerator
    :type gamma_next: float
    :param gamma_curr: Tempering parameter $\\gamma_{n-1}$ for the current distribution, will be used in the denominator
    :type gamma_curr: float
    :param overflow_mask: Mask indicating where there has been an overflow either in xnk or vnk. There we do not
                          compute a weight, it will be set to zero. Will have shape (N, T+1).
    :type overflow_mask: numpy.ndarray
    :return: Tuple containing `(W_unfolded, logw_unfolded, W_folded, logw_unfolded)`
    :rtype: tuple(numpy.array, numpy.array, numpy.array, numpy.array)
    """
    assert len(vnk.shape) == 3, "Velocities must be a 3-dimensional array."
    assert len(nlps.shape) == 2, "Negative log priors must be a 2-dimensional array."
    assert len(nlls.shape) == 2, "Negative log likelihoods must be a 2-dimensional array."
    N, Tplus1, d = vnk.shape

    ofm = overflow_mask.ravel()  # (N*(T+1), ) boolean mask, True if corresponding xnk or vnk is inf due to overflow
    ofm_seed = overflow_mask[:, 0].ravel()  # (N, ) same mask but only for seed particles

    log_num = np.ones(N*Tplus1)
    log_den = np.zeros(N)

    # Log numerator of the unfolded weights
    log_num[~ofm] = (-nlps.ravel()[~ofm]) + gamma_next*(-nlls.ravel()[~ofm])
    log_num[~ofm] -= 0.5*np.einsum('ij,ij->i', vnk.reshape(-1, d)[~ofm], vnk.reshape(-1, d)[~ofm].dot(inv_mass_next))
    log_num[~ofm] += 0.5*np.linalg.slogdet(inv_mass_next)[1]

    # Log Denominator of the unfolded weights
    log_den[~ofm_seed] = (-nlps[~ofm_seed, 0]) + gamma_curr*(-nlls[~ofm_seed, 0])
    log_den[~ofm_seed] -= 0.5*np.einsum(vnk[~ofm_seed, 0], vnk[~ofm_seed, 0].dot(inv_mass_curr))
    log_den[~ofm_seed] += 0.5*np.linalg.slogdet(inv_mass_curr)[1]

    # Unfolded weights
    logw_unfolded = log_num.reshape(N, Tplus1) - log_den[:, None]  # (N, T+1) log un-normalized unfolded weights
    W_unfolded = np.exp(logw_unfolded - logsumexp(logw_unfolded))  # (N, T+1) normalized unfolded weights

    # Folded weights
    logw_folded = logsumexp(logw_unfolded, axis=1) - np.log(Tplus1)  # (N, ) un-normalized folded weights
    W_folded = np.exp(logw_folded - logsumexp(logw_folded))  # (N, ) normalized folded weights

    return W_unfolded, logw_unfolded, W_folded, logw_folded


def estimate_with_cond_variance(xnk: NDArray, logw: NDArray, epsilons: NDArray, ss_dict: dict) -> NDArray:
    """Estimates a sufficient statistics using the conditional variance.

    Parameters
    ----------
    :param xnk: Snippet positions, has shape `(N, T+1, d)` where `d` is the dimension of the position space
    :type xnk: np.ndarray
    :param logw: Log un-normalized unfolded weights, has shape `(N, T+1)`
    :type logw: np.ndarray
    :param epsilons: Step sizes, one for each snippet, has shape `(N, )`
    :type epsilons: np.ndarray
    :param ss_dict: Dictionary of functions to compute the sufficient statistics. Keys should be strings and correspond
                    to the parameters of the epsilon distribution to be estimated, and values should be univariate
                    scalar-valued functions. For instance for an inverse gaussian distribution with fixed skewness
                    one should use `{'mean': lambda epsilon: epsilon}`
    :type ss_dict: dict
    :return: Dictionary of estimates for each sufficient statistics
    :rtype: dict
    """
    assert len(xnk.shape) == 3, "Snippet positions must have three dimensions `(N, T+1, d)`."
    assert len(logw.shape) == 2, "Log un-normalized unfolded weights must have two dimensions `(N, T+1)`."
    assert len(epsilons.shape) == 1, "Epsilons must be one dimensional `(N, )`."
    assert xnk.shape[0] == len(epsilons), "Number of snippets must match the number of epsilons."
    assert xnk.shape[:2] == logw.shape, "There must be a weight for each snippet position."

    # Weight computations are in common for all sufficient statistics
    T = logw.shape[1] - 1
    # Compute the discrete distribution mu(k | z, epsilon)
    mu_k_given_z_eps = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))  # (N, T+1)
    # Compute the conditional expectation for the position function mu(f| z, epsilon)
    cond_exp = np.sum(xnk * mu_k_given_z_eps[:, :, None], axis=1, keepdims=True)  # (N, 1, d)
    # Compute the squared norms
    norms = np.linalg.norm(xnk - cond_exp, axis=2)  # (N, T+1)
    # Base flag
    base_flag = (norms > 0) & (logw > -np.inf)

    estimators = {key: 0 for key in ss_dict.keys()}
    for param_name, ss in ss_dict.items():
        # Compute the test statistics for each snippet/epsilon
        T_hat = ss(epsilons)[:, None]   # (N, 1)
        # Compute terms of the form np.sum(sq_norms * w * T_eps) in a way to avoid numerical errors on the log scale
        T_hat_repeated = np.tile(T_hat, (T+1))  # (N, T+1)
        # Flag for when computation can be "logged"
        flag = base_flag & (T_hat_repeated != 0)  # (N, T+1) when computation is not zero
        # log of squared norms, computed only where computation is not zero
        log_sq_norms = 2*np.log(norms[flag])
        # Same for log weights and estimator
        logw_filtered = logw[flag]  # (n_filtered, )
        log_T_hat = np.log(T_hat_repeated[flag])    # (n_filtered, )
        # Compute scalar estimator
        estimator = np.exp(
            logsumexp(log_sq_norms + logw_filtered + log_T_hat) - logsumexp(log_sq_norms + logw_filtered)
        )
        estimators[param_name] = estimator
    return estimators


def hamiltonian_snippet(N: int, T: int, mass: NDArray, ESSrmin: float, sample_prior: callable,
                        compute_likelihoods_priors_gradients: callable, epsilon_params: dict,
                        adapt_mass: bool = False, verbose: bool = True, seed: Optional[int] = None):
    """Hamiltonian Snippets with Leapfrog integration and step size adaptation.

    Parameters
    ----------
    :param N: Number of particles
    :type N: int
    :param T: Number of integration steps
    :type T: int
    :param mass_diag: Diagonal of the mass matrix
    :type mass_diag: np.ndarray
    :param ESSrmin: Proportion of `N` that we target as our ESS when finding the next tempering parameter
    :type ESSrmin: float
    :param sample_prior: Function to sample from the prior, should take `N` and `rng` as arguments and return an array
    :type sample_prior: callable
    :param compute_likelihoods_priors_gradients: Function that takes positions of shape `(N, T+1, d)` as input and
                                                 computes the negative log-likelihood, its gradient, the negative
                                                 log prior and its gradient at these positions
    :type compute_likelihoods_priors_gradients: callable
    :param epsilon_params: Parameters for the distribution of the epsilons. Should be a dictionary containing
                           'distribution' which should be one of `['inv_gauss']` and parameters for them
    :type epsilon_params: dict
    :param adapt_mass: Whether to adapt the mass matrix diagonal or not
    :type adapt_mass: bool
    :param verbose: Whether to print progress of the algorithm
    :type verbose: bool
    :param seed: Seed for the random number generator
    :type seed: int or None
    """
    assert isinstance(N, int) and N >= 1, "Number of particles must be a positive integer."
    assert isinstance(T, int) and T >= 1, "Number of integration steps must be a positive integer."

    # Set up time-keeping, random number generation, printing, iterations, mass_matrix and more
    start_time = time.time()
    rng = np.random.default_rng(seed=seed if seed is not None else np.random.randint(low=0, high=10000000))
    verboseprint = print if verbose else lambda *a, **kwargs: None
    n = 1

    # Initialize particles, epsilons
    x = sample_prior(N, rng)
    d = x.shape[1]
    mass_curr = mass if mass is not None else np.eye(d)
    v = rng.normal(loc=0, scale=1, size=(N, d)).dot(np.linalg.cholesky(mass_curr).T)

    # Initial step sizes and mass matrix
    epsilons = sample_epsilons(eps_params=epsilon_params, N=N, rng=rng)

    # Storage
    epsilon_history = [epsilons]
    epsilon_params_history = [epsilon_params]  # parameters for the epsilon distribution
    gammas = [0.0]
    ess_history = [N]
    logLt = 0.0

    while gammas[n-1] < 1.0:
        verboseprint(f"Iteration {n} Gamma {gammas[n-1]: .3f} Epsilon {epsilon_params['to_print'].capitalize()}: {epsilon_params[epsilon_params['to_print']]: .4f}")

        # Construct trajectories
        xnk, vnk, nlps, nlls = leapfrog(x, v, T, epsilons, gammas[n-1], 1/mass_curr, compute_likelihoods_priors_gradients)
        verboseprint("\tTrajectories constructed.")

        # Check if there is any inf due to overflow error
        if not (np.all(np.isfinite(xnk)) and np.all(np.isfinite(vnk))):
            overflow_mask = np.any(~np.isfinite(xnk), axis=2) | np.any(~np.isfinite(vnk), axis=2)  # (N, T+1)
            verboseprint(f"\tOverflow Detected. Trajectories affected: {overflow_mask.any(axis=1).sum()}")
        else:
            overflow_mask = np.zeros((N, T+1), dtype=bool)

        # Select next tempering parameter based on target ESS
        gammas.append(next_annealing_param(gamma=gammas[n-1], ESSrmin=ESSrmin, llk=(-nlls[:, 0])))
        verboseprint(f"\tNext gamma selected: {gammas[-1]: .5f}")

        # Estimate new mass matrix diagonal using importance sampling
        mass_next = mass_curr

        # Compute weights and ESS
        W_unfolded, logw_unfolded, W_folded, logw_folded = compute_weights(
            vnk, nlps, nlls, np.linalg.inv(mass_next), np.linalg.inv(mass_curr), gammas[n], gammas[n-1], overflow_mask=overflow_mask)
        ess = 1 / np.sum(W_folded**2)  # folded ESS
        verboseprint(f"\tWeights Computed. Folded ESS {ess: .3f}")

        # Set the new cov matrix to the old one
        mass_curr = mass_next

        # Resample N particles out of N*(T+1) proportionally to unfolded weights
        A = rng.choice(a=N*(T+1), size=N, replace=True, p=W_unfolded.ravel())  # (N, )
        i_indices, k_indices = np.unravel_index(A, (N, T+1))  # (N, ) particles indices, (N, ) trajectory indices
        x = xnk[i_indices, k_indices]  # (N, d) resampled positions
        verboseprint(f"\tParticles resampled. PM {np.sum(k_indices > 0) / N: .3f}")

        # Refresh velocities
        v = rng.normal(loc=0, scale=1, size=(N, d)).dot(np.linalg.cholesky(mass_curr).T)
        verboseprint("\tVelocities refreshed.")

        # Compute log-normalizing constant estimates
        logLt += logsumexp(logw_folded) - np.log(N)
        verboseprint(f"\tLogLt {logLt}")

        # Step size adaptation
        xnk[overflow_mask] = 0.0
        epsilon_params.update(estimate_with_cond_variance(
            xnk=xnk, logw=logw_unfolded, epsilons=epsilons, ss_dict=epsilon_params['params_to_estimate']
        ))
        # step_size = estimate_new_epsilon_mean(xnk=xnk, logw=logw_unfolded, epsilons=epsilons, ss=lambda _eps: _eps)
        epsilons = sample_epsilons(eps_params=epsilon_params, N=N, rng=rng)
        verboseprint(f"\tEpsilon {epsilon_params['to_print'].capitalize()} {epsilon_params[epsilon_params['to_print']]}")

        # Storage
        epsilon_history.append(epsilons)
        epsilon_params_history.append(epsilon_params)
        ess_history.append(ess)

        n += 1
    runtime = time.time() - start_time
    return {"logLt": logLt, "gammas": gammas, "runtime": runtime, "epsilons": epsilon_history, "ess": ess_history,
            'epsilon_params_history': epsilon_params_history}


if __name__ == "__main__":
    # Load data and parameters
    grid_dim = 400
    params = create_log_cox_parameters(N=int(grid_dim**0.5))
    mass = params['metric_tensor']

    # Instantiate functions
    sample_prior = generate_sample_prior_function(dim=grid_dim, mu=params['mu'], l_covar=params['l_covar'])

    # Settings
    N = 500
    T = 30
    skewness = 0.5
    epsilon_params = {
        'distribution': 'inv_gauss',
        'skewness': skewness,
        'mean': 0.1,
        'params_to_estimate': {'mean': lambda epsilon: epsilon},
        'to_print': 'mean'
    }

    out = hamiltonian_snippet(N=N, T=T, mass=mass, ESSrmin=0.8, sample_prior=sample_prior,
                              compute_likelihoods_priors_gradients=lambda x: nlp_gnlp_nll_and_gnll(x, params),
                              epsilon_params=epsilon_params,
                              adapt_mass=False, verbose=True, seed=1234)


