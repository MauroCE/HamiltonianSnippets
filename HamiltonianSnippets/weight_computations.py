import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from typing import Tuple


def compute_weights(
        vnk: NDArray,
        nlps: NDArray,
        nlls: NDArray,
        inv_mass_diag_next: NDArray,
        inv_mass_diag_curr: NDArray,
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
    :param inv_mass_diag_next: Diagonal of the inverse mass matrix at time `n`
    :type inv_mass_diag_next: numpy.ndarray
    :param inv_mass_diag_curr: Diagonal of the inverse mass matrix at time `n-1`
    :type inv_mass_diag_curr: numpy.ndarray
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

    log_num = np.repeat(-np.inf, N*Tplus1)  # np.ones(N*Tplus1)
    log_den = np.zeros(N)

    # Log numerator of the unfolded weights
    log_num[~ofm] = (-nlps.ravel()[~ofm]) + gamma_next*(-nlls.ravel()[~ofm])
    log_num[~ofm] -= 0.5*np.sum(inv_mass_diag_next * vnk.reshape(-1, d)[~ofm]**2, axis=1)
    log_num[~ofm] += 0.5*np.sum(np.log(inv_mass_diag_next))

    # Log Denominator of the unfolded weights
    log_den[~ofm_seed] = (-nlps[~ofm_seed, 0]) + gamma_curr*(-nlls[~ofm_seed, 0])
    log_den[~ofm_seed] -= 0.5*np.sum((vnk[~ofm_seed, 0]**2) * inv_mass_diag_curr, axis=1)
    log_den[~ofm_seed] += 0.5*np.sum(np.log(inv_mass_diag_curr))

    # Unfolded weights
    logw_unfolded = log_num.reshape(N, Tplus1) - log_den[:, None]  # (N, T+1) log un-normalized unfolded weights
    W_unfolded = np.exp(logw_unfolded - logsumexp(logw_unfolded))  # (N, T+1) normalized unfolded weights

    # Overflown should lead to zero weights
    assert W_unfolded[overflow_mask].sum() == 0, "Weights should be zero."

    # Folded weights
    logw_folded = logsumexp(logw_unfolded, axis=1) - np.log(Tplus1)  # (N, ) un-normalized folded weights
    W_folded = np.exp(logw_folded - logsumexp(logw_folded))  # (N, ) normalized folded weights

    return W_unfolded, logw_unfolded, W_folded, logw_folded
