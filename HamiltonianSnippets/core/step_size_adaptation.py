import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import invgauss
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'STIXGeneral'})


def estimate_with_cond_variance(xnk: NDArray, logw_criterion: NDArray, epsilons: NDArray, epsilon_params: dict, skip_overflown: bool,
                                overflow_mask: NDArray, plot_Q_criterion: bool) -> NDArray:
    """Estimates a sufficient statistics using the conditional variance.

    Parameters
    ----------
    :param xnk: Snippet positions, has shape `(N, T+1, d)` where `d` is the dimension of the position space
    :type xnk: np.ndarray
    :param logw_criterion: Log un-normalized criterion weights, has shape `(N, T+1)`, where are not the unfolded weights
    :type logw_criterion: np.ndarray
    :param epsilons: Step sizes, one for each snippet, has shape `(N, )`
    :type epsilons: np.ndarray
    :param epsilon_params: Dictionary of epsilon parameters
    :type epsilon_params: dict
    :param skip_overflown: Whether to skip the trajectories that had an epsilon that lead to overflow. Shape (N, )
    :type skip_overflown: bool
    :param overflow_mask: Mask determining if the snippet has encountered an overflow error
    :type overflow_mask: np.ndarray
    :param plot_Q_criterion: Whether to plot the Q criterion
    :type plot_Q_criterion: bool
    :return: Dictionary of estimates for each sufficient statistics
    :rtype: dict
    """
    assert len(xnk.shape) == 3, "Snippet positions must have three dimensions `(N, T+1, d)`."
    assert len(logw_criterion.shape) == 2, "Log un-normalized unfolded weights must have two dimensions `(N, T+1)`."
    assert len(epsilons.shape) == 1, "Epsilons must be one dimensional `(N, )`."
    assert xnk.shape[0] == len(epsilons), "Number of snippets must match the number of epsilons."
    assert xnk.shape[:2] == logw_criterion.shape, "There must be a weight for each snippet position."

    # Weight computations are in common for all sufficient statistics
    N, Tplus1 = logw_criterion.shape
    ofm = overflow_mask if skip_overflown else np.zeros(N, dtype=bool)

    # Compute the discrete distribution mu(k | z, epsilon)
    mu_k_given_z_eps = np.exp(logw_criterion[~ofm] - logsumexp(logw_criterion[~ofm], axis=1, keepdims=True))  # (N, T+1)
    # Compute the conditional expectation for the position function mu(f| z, epsilon)
    cond_exp = np.sum(xnk[~ofm] * mu_k_given_z_eps[:, :, None], axis=1, keepdims=True)  # (N, 1, d)
    # Compute the squared norms
    norms = np.linalg.norm(xnk[~ofm] - cond_exp, axis=2)  # (N, T+1)
    # Base flag
    base_flag = (norms > 0) & (logw_criterion[~ofm] > -np.inf)

    if plot_Q_criterion and base_flag.all():
        Q = np.sum((np.exp(logw_criterion - logsumexp(logw_criterion))*norms**2).reshape(N, Tplus1), axis=1)  # (N, )
        sort_ix = np.argsort(epsilons)
        fig, ax = plt.subplots()
        ax.plot(epsilons[sort_ix], Q[sort_ix], marker='o', color='lightcoral', markeredgecolor='firebrick', lw=2, label=r"$\mathregular{\mathcal{Q}}$")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        plt.show()

    estimators = {key: 0 for key in epsilon_params['params_to_estimate'].keys()}
    if epsilon_params['distribution'] != "discrete_uniform":
        for param_name, ss in epsilon_params['params_to_estimate'].items():
            # Compute the test statistics for each snippet/epsilon
            T_hat = ss(epsilons[~ofm])[:, None]   # (N, 1)
            # Compute terms of the form np.sum(sq_norms * w * T_eps) in a way to avoid numerical errors on the log scale
            T_hat_repeated = np.tile(T_hat, Tplus1)  # (N, T+1)
            # Flag for when computation can be "logged"
            flag = base_flag & (T_hat_repeated != 0)  # (N, T+1) when computation is not zero
            # log of squared norms, computed only where computation is not zero
            log_sq_norms = 2*np.log(norms[flag])
            # Same for log weights and estimator
            logw_filtered = logw_criterion[~ofm][flag]  # (n_filtered, )
            log_T_hat = np.log(T_hat_repeated[flag])    # (n_filtered, )
            # Compute scalar estimator
            estimator = np.exp(
                logsumexp(log_sq_norms + logw_filtered + log_T_hat) - logsumexp(log_sq_norms + logw_filtered)
            )
            estimators[param_name] = estimator
    else:
        estimators = {'values': epsilon_params['values']}
    return estimators


def sample_epsilons(eps_params: dict, N: int, rng: np.random.Generator) -> NDArray:
    """Samples epsilons according to the correct distribution. Currently, only two options are available, either an
    inverse gaussian with a fixed skewness or a discrete uniform distribution.

    Parameters
    ----------
    :param eps_params: Step size parameters
    :type eps_params: dict
    :param N: Number of step sizes to sample, should be the same as the number of particles
    :type N: int
    :param rng: Random number generator for reproducibility
    :type: np.random.Generator
    :return: Array of sampled step sizes of shape (N, )
    :rtype: np.ndarray
    """
    assert 'distribution' in eps_params, "Epsilon parameters must contain a key 'distribution'."
    match eps_params['distribution']:

        case 'inv_gauss':
            assert eps_params['skewness'] > 0, "Skewness must be strictly positive for an inverse gaussian."
            assert eps_params['mean'] > 0, "Mean must be strictly positive for an inverse gaussian."
            lambda_param = 9*eps_params['mean'] / eps_params['skewness']**2  # since skewness = 3*sqrt(mu/lambda)
            return invgauss.rvs(mu=eps_params['mean']/lambda_param, loc=0, scale=lambda_param, size=N, random_state=rng)

        case 'discrete_uniform':
            assert len(eps_params['values']) > 0, "When using a discrete uniform distribution, there should be at least one value, one for each group."
            assert np.all(np.array(eps_params['values']) > 0), "Epsilon values must be positive for all groups when using a uniform discrete distribution."
            return rng.choice(a=eps_params['values'], size=N, replace=True)
