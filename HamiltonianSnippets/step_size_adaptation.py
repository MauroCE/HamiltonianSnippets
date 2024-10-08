import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import invgauss


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


def sample_epsilons(eps_params: dict, N: int, rng: np.random.Generator) -> NDArray:
    """Samples epsilons according to the correct distribution."""
    assert 'distribution' in eps_params, "Epsilon parameters must contain a key 'distribution'."
    match eps_params['distribution']:
        case 'inv_gauss':
            assert eps_params['skewness'] > 0, "Skewness must be strictly positive for an inverse gaussian."
            assert eps_params['mean'] > 0, "Mean must be strictly positive for an inverse gaussian."
            lambda_param = 9*eps_params['mean'] / eps_params['skewness']**2  # since skewness = 3*sqrt(mu/lambda)
            return invgauss.rvs(mu=eps_params['mean']/lambda_param, loc=0, scale=lambda_param, size=N, random_state=rng)


