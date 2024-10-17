import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from typing import Tuple


def compute_weights_new(
        vnk: NDArray,
        nlps: NDArray,
        nlls: NDArray,
        mass_params: dict,
        gamma_next: float,
        gamma_curr: float,
        overflow_mask: NDArray,
        computation_for_mass_matrix_adaptation: bool = False
        ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Computes unfolded and folded weights.

    Parameters
    ----------
    :param vnk: Velocities constructed with $\\psi_{n-1}$, has shape `(N, T+1, d)` where `d` is the velocity dimension
    :type vnk: numpy.ndarray
    :param nlps: Negative log prior density on the snippets, has shape `(N, T+1)`
    :type nlps: numpy.ndarray
    :param nlls: Negative log likelihood on the snippets, has shape `(N, T+1)`
    :type nlls: numpy.ndarray
    :param mass_params: Mass matrix parameters
    :type mass_params: dict
    :param gamma_next: Tempering parameter $\\gamma_n$ for the next distribution, will be used in the numerator
    :type gamma_next: float
    :param gamma_curr: Tempering parameter $\\gamma_{n-1}$ for the current distribution, will be used in the denominator
    :type gamma_curr: float
    :param overflow_mask: Mask indicating where there has been an overflow either in xnk or vnk. There we do not
                          compute a weight, it will be set to zero. Will have shape (N, T+1).
    :type overflow_mask: numpy.ndarray
    :param computation_for_mass_matrix_adaptation: If True, we use varpi_{n-1} in the numerator too
    :type computation_for_mass_matrix_adaptation: bool
    :return: Tuple containing `(W_unfolded, logw_unfolded, W_folded, logw_unfolded, logw_criterion)`
    :rtype: tuple(numpy.array, numpy.array, numpy.array, numpy.array, numpy.array)
    """
    assert len(vnk.shape) == 3, "Velocities must be a 3-dimensional array."
    assert len(nlps.shape) == 2, "Negative log priors must be a 2-dimensional array."
    assert len(nlls.shape) == 2, "Negative log likelihoods must be a 2-dimensional array."
    N, Tplus1, d = vnk.shape

    # First part is True when corresponding xnk or vnk is inf due to overflow
    # Second part is True when squaring vnk leads to overflow
    # max_invmass = inv_mass_diag_next.max()  # maximum of the inverse mass diagonal
    ofm = overflow_mask.ravel() | np.any(np.abs(vnk.reshape(-1, d)) >= np.sqrt(np.finfo(np.float64).max), axis=1)  # (N*(T+1), )
    # same mask but only for seed particles
    ofm_seed = overflow_mask[:, 0].ravel() | np.any(np.abs(vnk[:, 0]) >= np.sqrt(np.finfo(np.float64).max), axis=1)  # (N, )

    log_num = np.repeat(-np.inf, N*Tplus1)  # default to zero denominator
    log_num_unfolded = np.repeat(-np.inf, N*Tplus1)
    log_num_criterion = np.repeat(-np.inf, N*Tplus1)
    log_den = np.repeat(np.nan, N)  # no overflown particle can ever become a seed

    # Log numerator of the unfolded weights
    log_num[~ofm] = (-nlps.ravel()[~ofm])
    log_num[~ofm] += compute_quadratic_form(mass_params, vnk.reshape(-1, d)[~ofm], numerator_computation=not computation_for_mass_matrix_adaptation)  #-= 0.5*np.sum(inv_mass_diag_next * vnk.reshape(-1, d)[~ofm]**2, axis=1)
    log_num[~ofm] += compute_determinant_term(mass_params, numerator_computation=not computation_for_mass_matrix_adaptation)  # 0.5*np.sum(np.log(inv_mass_diag_next))

    log_num_unfolded[~ofm] = log_num[~ofm] + gamma_next*(-nlls.ravel()[~ofm])
    log_num_criterion[~ofm] = log_num[~ofm] + gamma_curr*(-nlls.ravel()[~ofm])

    # Log Denominator of the unfolded weights
    log_den[~ofm_seed] = (-nlps[~ofm_seed, 0]) + gamma_curr*(-nlls[~ofm_seed, 0])
    log_den[~ofm_seed] += compute_quadratic_form(mass_params, vnk[~ofm_seed, 0], numerator_computation=False)#-= 0.5*np.sum((vnk[~ofm_seed, 0]**2) * inv_mass_diag_curr, axis=1)
    log_den[~ofm_seed] += compute_determinant_term(mass_params, numerator_computation=False)  # 0.5*np.sum(np.log(inv_mass_diag_curr))

    # Unfolded weights
    logw_unfolded = log_num_unfolded.reshape(N, Tplus1) - log_den[:, None]  # (N, T+1) log un-normalized unfolded weights
    W_unfolded = np.exp(logw_unfolded - logsumexp(logw_unfolded))  # (N, T+1) normalized unfolded weights

    # Weights for criterion
    logw_criterion = log_num_criterion.reshape(N, Tplus1) - log_den[:, None]  # (N, T+1) similar to unfolded

    # Overflown should lead to zero weights
    if W_unfolded[overflow_mask].sum() != 0:
        raise ValueError(f"Weights should be zero but they are {W_unfolded[overflow_mask].sum()}")

    # Folded weights
    logw_folded = logsumexp(logw_unfolded, axis=1) - np.log(Tplus1)  # (N, ) un-normalized folded weights
    W_folded = np.exp(logw_folded - logsumexp(logw_folded))  # (N, ) normalized folded weights

    return W_unfolded, logw_unfolded, W_folded, logw_folded, logw_criterion


def compute_quadratic_form(mass_params: dict, v: NDArray, numerator_computation: bool = True):
    """Computes the quadratic form for the unfolded weights. v must be masked already and sliced."""
    suffix = "next" if numerator_computation else "curr"
    match mass_params["strategy"], mass_params["matrix_type"]:
        case ("fixed" | "schedule" | "adaptive"), "diag":
            return -0.5*np.sum(mass_params[f"mass_diag_{suffix}"] * v**2, axis=1)
        case ("fixed" | "schedule"), "full":
            M_inv_v = np.linalg.solve(mass_params[f"mass_{suffix}"], v.T).T
            return -0.5*np.einsum('ij,ij->i', v, M_inv_v)
        case "adaptive", "full":
            raise NotImplemented("Mass Matrix adaptation with a full mass matrix is not implemented yet.")


def compute_determinant_term(mass_params: dict, numerator_computation: bool):
    """Computes the determinant term."""
    suffix = "next" if numerator_computation else "curr"
    match mass_params['strategy'], mass_params['matrix_type']:
        case "fixed", ("diag" | "full"):
            return 0  # numerator and denominator determinant terms would cancel out
        case ("schedule", ("diag" | "full")) | ("adaptive", "diag"):
            return -0.5*mass_params[f"log_det_mass_{suffix}"]
        case ("adaptive", "full"):
            raise NotImplemented("Mass Matrix adaptation with a full mass matrix is not implemented yet.")


def compute_weights(
        vnk: NDArray,
        nlps: NDArray,
        nlls: NDArray,
        inv_mass_diag_next: NDArray,
        inv_mass_diag_curr: NDArray,
        gamma_next: float,
        gamma_curr: float,
        overflow_mask: NDArray,
        ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
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
    :return: Tuple containing `(W_unfolded, logw_unfolded, W_folded, logw_unfolded, logw_criterion)`
    :rtype: tuple(numpy.array, numpy.array, numpy.array, numpy.array, numpy.array)
    """
    assert len(vnk.shape) == 3, "Velocities must be a 3-dimensional array."
    assert len(nlps.shape) == 2, "Negative log priors must be a 2-dimensional array."
    assert len(nlls.shape) == 2, "Negative log likelihoods must be a 2-dimensional array."
    N, Tplus1, d = vnk.shape

    # ofm = overflow_mask.ravel()  # (N*(T+1), ) boolean mask, True if corresponding xnk or vnk is inf due to overflow
    # ofm_seed = overflow_mask[:, 0].ravel()  # (N, ) same mask but only for seed particles

    # First part is True when corresponding xnk or vnk is inf due to overflow
    # Second part is True when squaring vnk leads to overflow
    max_invmass = inv_mass_diag_next.max()  # maximum of the inverse mass diagonal
    ofm = overflow_mask.ravel() | np.any(np.abs(vnk.reshape(-1, d)) >= np.sqrt(np.finfo(np.float64).max), axis=1)  # (N*(T+1), )
    # same mask but only for seed particles
    ofm_seed = overflow_mask[:, 0].ravel() | np.any(np.abs(vnk[:, 0]) >= np.sqrt(np.finfo(np.float64).max), axis=1)  # (N, )

    log_num = np.repeat(-np.inf, N*Tplus1)  # default to zero denominator
    log_num_unfolded = np.repeat(-np.inf, N*Tplus1)
    log_num_criterion = np.repeat(-np.inf, N*Tplus1)
    log_den = np.repeat(np.nan, N)  # no overflown particle can ever become a seed

    # Log numerator of the unfolded weights
    log_num[~ofm] = (-nlps.ravel()[~ofm])  # + gamma_next*(-nlls.ravel()[~ofm])
    log_num[~ofm] -= 0.5*np.sum(inv_mass_diag_next * vnk.reshape(-1, d)[~ofm]**2, axis=1)
    log_num[~ofm] += 0.5*np.sum(np.log(inv_mass_diag_next))

    log_num_unfolded[~ofm] = log_num[~ofm] + gamma_next*(-nlls.ravel()[~ofm])
    log_num_criterion[~ofm] = log_num[~ofm] + gamma_curr*(-nlls.ravel()[~ofm])

    # Log Denominator of the unfolded weights
    log_den[~ofm_seed] = (-nlps[~ofm_seed, 0]) + gamma_curr*(-nlls[~ofm_seed, 0])
    log_den[~ofm_seed] -= 0.5*np.sum((vnk[~ofm_seed, 0]**2) * inv_mass_diag_curr, axis=1)
    log_den[~ofm_seed] += 0.5*np.sum(np.log(inv_mass_diag_curr))

    # Unfolded weights
    logw_unfolded = log_num_unfolded.reshape(N, Tplus1) - log_den[:, None]  # (N, T+1) log un-normalized unfolded weights
    W_unfolded = np.exp(logw_unfolded - logsumexp(logw_unfolded))  # (N, T+1) normalized unfolded weights

    # Weights for criterion
    logw_criterion = log_num_criterion.reshape(N, Tplus1) - log_den[:, None]  # (N, T+1) similar to unfolded

    # Overflown should lead to zero weights
    if W_unfolded[overflow_mask].sum() != 0:
        raise ValueError(f"Weights should be zero but they are {W_unfolded[overflow_mask].sum()}")

    # Folded weights
    logw_folded = logsumexp(logw_unfolded, axis=1) - np.log(Tplus1)  # (N, ) un-normalized folded weights
    W_folded = np.exp(logw_folded - logsumexp(logw_folded))  # (N, ) normalized folded weights

    return W_unfolded, logw_unfolded, W_folded, logw_folded, logw_criterion
