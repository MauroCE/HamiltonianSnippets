import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


def conditional_variance_criterion(xnk: NDArray, logw: NDArray, epsilons: NDArray, ss: callable):
    """Conditional variance criterion used for adaptation of the Leapfrog step size.

    Parameters
    ----------
    :param xnk: Snippet positions, has shape `(N, T+1, d)` where `d` is the dimension of the position space
    :type xnk: np.ndarray
    :param logw: Log un-normalized unfolded weights, has shape `(N, T+1)`
    :type logw: np.ndarray
    :param epsilons: Step sizes, one for each snippet, has shape `(N, )`
    :type epsilons: np.ndarray
    :param ss: Function to compute the sufficient statistics, must be a univariate scalar-valued function
    :type ss: callable
    """
    assert len(xnk.shape) == 3, "Snippet positions must have three dimensions `(N, T+1, d)`."
    assert len(logw.shape) == 2, "Log un-normalized unfolded weights must have two dimensions `(N, T+1)`."
    assert len(epsilons.shape) == 1, "Epsilons must be one dimensional `(N, )`."
    assert xnk.shape[0] == len(epsilons), "Number of snippets must match the number of epsilons."
    assert xnk.shape[:2] == logw.shape, "There must be a weight for each snippet position."

    T = logw.shape[1] - 1
    # Compute the discrete distribution mu(k | z, epsilon)
    mu_k_given_z_eps = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))  # (N, T+1)
    # Compute the conditional expectation for the position function mu(f| z, epsilon)
    cond_exp = np.sum(xnk * mu_k_given_z_eps[:, :, None], axis=1, keepdims=True)  # (N, 1, d)
    # Compute the squared norms
    sq_norms = np.linalg.norm(xnk - cond_exp, axis=2)**2  # (N, T+1)
    # Compute the test statistics for each snippet/epsilon
    T_eps = ss(epsilons)[:, None]   # (N, 1)
    # Compute terms of the form np.sum(sq_norms * w * T_eps) in a way to avoid numerical errors on the log scale
    T_eps_tiled = np.tile(T_eps, (T+1))  # (N, T+1)
    flag = (sq_norms > 0) & (logw > -np.inf) & (T_eps_tiled > 0)  # (N, T+1) when computation is not zero
    log_sq_norms = 2*np.log(np.linalg.norm(xnk - cond_exp, axis=2)[flag])  # (n_filtered, )
    logw_filtered = logw[flag]               # (n_filtered, )
    log_T_eps = np.log(T_eps_tiled[flag])    # (n_filtered, )
    return np.exp(
        logsumexp(log_sq_norms + logw_filtered + log_T_eps) - logsumexp(log_sq_norms + logw_filtered)
    )  # scalar, conditional variance
