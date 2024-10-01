import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.linalg import norm
from typing import Tuple


def compute_weights_and_ess(
        vnk: NDArray,
        nlps: NDArray,
        nlls: NDArray,
        inv_mass_diag_next: NDArray,
        inv_mass_diag_curr: NDArray,
        gamma_next: float,
        gamma_curr: float,
        ) -> Tuple[NDArray, NDArray, NDArray, NDArray, np.float]:
    """Computes unfolded weights for an identity mass matrix.

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
    :return: Tuple containing `(W_unfolded, logw_unfolded, W_folded, logw_unfolded, folded_ess)`
    :rtype: tuple(numpy.array, numpy.array, numpy.array, numpy.array, float)
    """
    assert len(vnk.shape) == 3, "Velocities must be a 3-dimensional array."
    assert len(nlps.shape) == 2, "Negative log priors must be a 2-dimensional array."
    assert len(nlls.shape) == 2, "Negative log likelihoods must be a 2-dimensional array."
    N, Tplus1, d = vnk.shape

    # Log numerator of the unfolded weights
    log_num = (-nlps) + gamma_next*(-nlls)  # (N, T+1)
    log_num -= 0.5*np.sum(inv_mass_diag_next * vnk.reshape(-1, d)**2, axis=1).reshape(N, Tplus1)  # (N, T+1)
    log_num += 0.5*np.sum(np.log(inv_mass_diag_next))  # (N, T+1)

    # Log denominator of the unfolded weights
    log_den = (-nlps[:, 0]) + gamma_curr*(-nlls[:, 0])  # (N, )
    log_den -= 0.5*np.sum((vnk**2) * inv_mass_diag_curr, axis=1)  # (N, )
    log_den += 0.5*np.sum(np.log(inv_mass_diag_curr))  # (N, )

    # Unfolded weights
    logw_unfolded = log_num - log_den[:, None]  # (N, T+1) log un-normalized unfolded weights
    W_unfolded = np.exp(logw_unfolded - logsumexp(logw_unfolded))  # (N, T+1) normalized unfolded weights

    # Folded weights
    logw_folded = logsumexp(logw_unfolded, axis=1) - np.log(Tplus1)  # (N, ) un-normalized folded weights
    W_folded = np.exp(logw_folded - logsumexp(logw_folded))  # (N, ) normalized folded weights

    # Folded ESS
    ess = 1 / np.sum(W_folded**2)

    return W_unfolded, logw_unfolded, W_folded, logw_folded, ess
