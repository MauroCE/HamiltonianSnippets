import numpy as np
from .weight_computations import compute_weights


def adapt_mass_matrix(mass_params, xnk, vnk, nlps, nlls, gammas, n, overflow_mask):
    """No idea."""
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
            # mass_params['inv_mass_diag_next'] = 1 / mass_params['mass_diag_next']  # Inverse(Mass Matrix)
        case "schedule", "full":
            mass_params['mass_next'] = mass_params['schedule_func'](gammas[n])
            mass_params['chol_mass_next'] = np.linalg.cholesky(mass_params['mass_next'])
            mass_params['log_det_mass_next'] = np.linalg.slogdet(mass_params['mass_next']).logabsdet
        case "adaptive", "full":
            raise NotImplemented("Mass Matrix adaptation with a full mass matrix is not implemented yet.")
    return mass_params
