import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.linalg import norm
from typing import Tuple


def compute_weights_and_ess(log_num: NDArray, log_den: NDArray):
    """Computes folded and unfolded weights from the log numerator and log denominator of unfolded weights.

    Parameters
    ----------
    :param log_num: Log numerator of the unfolded weights, has shape `(N, T+1)`
    :type log_num: numpy.ndarray
    :param log_den: Log denominator of the unfolded weights, has shape `(N, )`
    :type log_den: numpy.ndarray
    """
    assert len(log_num.shape) == 2, "Log numerator must be a 2-dimensional array."
    assert len(log_den.shape) == 1, "Log denominator must be a 1-dimensional array."
    assert log_num.shape[0] == len(log_den), "First dimension of log numerator and length of log denominator must match"
    T = log_num.shape[1] - 1

    # Unfolded weights
    logw_unfolded = log_num - log_den[:, None]  # (N, T+1) log un-normalized unfolded weights
    W_unfolded = np.exp(logw_unfolded - logsumexp(logw_unfolded))  # (N, T+1) normalized unfolded weights

    # Folded weights
    logw_folded = logsumexp(logw_unfolded, axis=1) - np.log(T+1)  # (N, ) un-normalized folded weights
    W_folded = np.exp(logw_folded - logsumexp(logw_folded))  # (N, ) normalized folded weights

    # Folded ESS
    ess = 1 / np.sum(W_folded**2)

    return W_unfolded, logw_unfolded, W_folded, logw_folded, ess


def compute_unfolded_weights_identity_mass_matrix(vnk: NDArray, nlps: NDArray, nlls: NDArray, gamma_next: float,
                                                  gamma_curr: float) -> Tuple[NDArray, NDArray, NDArray,
                                                                              NDArray, np.float]:
    """Computes unfolded weights for an identity mass matrix.

    Parameters
    ----------
    :param vnk: Velocities constructed with $\\psi_{n-1}$, has shape `(N, T+1, 2)` where `d` is the velocity dimension
    :type vnk: numpy.ndarray
    :param nlps: Negative log prior density on the snippets, has shape `(N, T+1)`
    :type nlps: numpy.ndarray
    :param nlls: Negative log likelihood on the snippets, has shape `(N, T+1)`
    :type nlls: numpy.ndarray
    :param gamma_next: Tempering parameter $\\gamma_n$ for the next distribution, will be used in the numerator
    :type gamma_next: float
    :param gamma_curr: Tempering parameter $\\gamma_{n-1}$ for the current distribution, will be used in the denominator
    :type gamma_curr: float
    :return: Tuple containing `(W_unfolded, logw_unfolded, W_folded, logw_unfolded, folded_ess)`
    :rtype: tuple(numpy.array, numpy.array, numpy.array, numpy.array, float)
    """
    # Log numerator and log denominator of the unfolded weights
    log_num = (-nlps) + gamma_next*(-nlls) - 0.5*norm(vnk, axis=2)**2  # (N, T+1) log numerators
    log_den = (-nlps[:, 0]) + gamma_curr*(-nlls[:, 0]) - 0.5*norm(vnk[:, 0], axis=1)**2  # (N, ) log denominators
    return compute_weights_and_ess(log_num=log_num, log_den=log_den)
