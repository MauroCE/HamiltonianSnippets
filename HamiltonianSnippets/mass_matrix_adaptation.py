import numpy as np
from .weight_computations import compute_weights
from numpy.typing import NDArray


def update_mass_matrix(mass_params: dict, xnk: NDArray, vnk: NDArray, nlps: NDArray, nlls: NDArray, gammas: list,
                       n: int, overflow_mask: NDArray) -> dict:
    """Selects the next mass matrix either using adaptation, a pre-defined schedule or keeping it fixed.

    Parameters
    ----------
    :param mass_params: Mass matrix parameters
    :type mass_params: dict
    :param xnk: Positions of shape (N, T+1, d)
    :type xnk: np.ndarray
    :param vnk: Velocities of shape (N, T+1, d)
    :type vnk: np.ndarray
    :param nlps: Negative log priors of shape (N, T+1)
    :type nlps: np.ndarray
    :param nlls: Negative log likelihoods of shape (N, T+1)
    :type nlls: np.ndarray
    :param gammas: Tempering parameters
    :type gammas: list
    :param n: Index of the current iteration should be between 1 and P
    :type n: int
    :param overflow_mask: Boolean mask of where computations for positions or velocities overflew
    :type overflow_mask: np.ndarray
    :return: Updated mass matrix parameters
    :rtype: dict
    """
    N, Tplus1, d = xnk.shape
    match mass_params["strategy"], mass_params["matrix_type"]:
        case "fixed", ("diag" | "full"):
            return mass_params  # do nothing
        case ("schedule" | "adaptive"), "diag":
            if mass_params['strategy'] == "schedule":
                mass_params['mass_diag_next'] = mass_params['schedule_func'](gammas[n])  # Mass Matrix diagonal
            else:
                W_mass_est, _, _, _, _ = compute_weights(
                    vnk=vnk, nlps=nlps, nlls=nlls, mass_params=mass_params, gamma_next=gammas[n], gamma_curr=gammas[n-1],
                    computation_for_mass_matrix_adaptation=True,
                    overflow_mask=overflow_mask
                )
                weighted_mean = np.average(xnk.reshape(-1, d)[~overflow_mask.ravel()], axis=0, weights=W_mass_est.ravel()[~overflow_mask.ravel()])  # (d, )
                mass_params['mass_diag_next'] = 1 / np.average((xnk.reshape(-1, d)[~overflow_mask.ravel()] - weighted_mean)**2, axis=0, weights=W_mass_est.ravel()[~overflow_mask.ravel()])
            mass_params['chol_mass_diag_next'] = np.sqrt(mass_params['mass_diag_next'])  # Cholesky(Mass Matrix)
            mass_params['log_det_mass_next'] = np.sum(np.log(mass_params['mass_diag_next']))  # Log determinant of the mass matrix
        case "schedule", "full":
            mass_params['mass_next'] = mass_params['schedule_func'](gammas[n])
            mass_params['chol_mass_next'] = np.linalg.cholesky(mass_params['mass_next'])
            mass_params['log_det_mass_next'] = np.linalg.slogdet(mass_params['mass_next']).logabsdet
        case "adaptive", "full":
            raise NotImplemented("Mass Matrix adaptation with a full mass matrix is not implemented yet.")
    return mass_params
