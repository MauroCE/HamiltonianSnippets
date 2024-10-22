import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def leapfrog(x: NDArray, v: NDArray, T: int, epsilons: NDArray, gamma_curr: float, mass_params: dict,
             compute_likelihoods_priors_gradients: callable) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Leapfrog integration.

    Parameters
    ----------
    :param x: Positions of shape (N, d)
    :type x: np.ndarray
    :param v: Velocities of shape (N, d)
    :type v: np.ndarray
    :param T: Number of leapfrog steps
    :type T: int
    :param epsilons: Step sizes, one for each position-velocity pair
    :type epsilons: np.ndarray
    :param gamma_curr: Current tempering parameter, typically \\gamma_{n-1}
    :type gamma_curr: float
    :param mass_params: Parameters for the mass matrix
    :type mass_params: dict
    :param compute_likelihoods_priors_gradients: Function computing nlps, gnlps, nlls and gnlls
    :type compute_likelihoods_priors_gradients: callable
    :return: Positions (N, T+1, d), Velocities (N, T+1, d), Neg Log Priors (N, T+1), Neg Log Likelihood (N, T+1)
    :type: tuple
    """
    N, d = x.shape
    if len(epsilons.shape) == 1:
        epsilons = epsilons[:, None]

    # Select momentum update function
    def momentum_update_when_gamma_is_zero(gnlp, gnll):
        """Computes momentum update when gamma=0, meaning gradient of negative log likelihood is not used."""
        return epsilons * gnlp.reshape(gnll.shape)  # reshape used only for pylint

    def momentum_update_when_gamma_larger_than_zero(gnlp, gnll):
        """Computes momentum update when gamma>0, meaning gradient of negative log likelihood is used."""
        return epsilons*(gnlp + gamma_curr*gnll)
    mom_update = momentum_update_when_gamma_is_zero if gamma_curr == 0 else momentum_update_when_gamma_larger_than_zero

    # Initialize snippets, negative log priors and negative log likelihoods
    xnk = np.full((N, T+1, d), np.nan)
    vnk = np.full((N, T+1, d), np.nan)
    nlps = np.full((N, T+1), np.nan)  # negative log priors
    nlls = np.full((N, T+1), np.nan)  # negative log likelihoods
    xnk[:, 0] = x  # seed positions
    vnk[:, 0] = v  # seed velocities

    # First momentum half-step
    nlps[:, 0], gnlps, nlls[:, 0], gnlls = compute_likelihoods_priors_gradients(x)
    v = v - 0.5*mom_update(gnlp=gnlps, gnll=gnlls)  # (N, d)

    # T - 1 position and velocity full steps
    for k in range(T-1):

        # Full position step
        x = x + epsilons*inv_mass_times_v(v, mass_params)

        # Full momentum step
        nlps[:, k+1], gnlps, nlls[:, k+1], gnlls = compute_likelihoods_priors_gradients(x)
        v = v - mom_update(gnlp=gnlps, gnll=gnlls)

        # Store positions and velocities
        vnk[:, k+1] = v
        xnk[:, k+1] = x

    # Final position half-step
    x = x + epsilons*inv_mass_times_v(v, mass_params)

    # Final momentum half-step
    nlps[:, -1], gnlps, nlls[:, -1], gnlls = compute_likelihoods_priors_gradients(x)
    v = v - 0.5*mom_update(gnlp=gnlps, gnll=gnlls)

    # Store final position and velocity
    xnk[:, -1] = x
    vnk[:, -1] = v

    return xnk, vnk, nlps, nlls


def inv_mass_times_v(v: NDArray, mass_params: dict):
    """Computes M^{-1} v for the various scenarios.

    Parameters
    ----------
    :param v: Velocities to multiply by the inverse mass matrix
    :type v: np.ndarray
    :param mass_params: Parameters for the mass matrix
    :type mass_params: dict
    :return: Inverse mass matrix times v, an array of shape (N, d)
    :rtype: np.ndarray
    """
    if mass_params['matrix_type'] == "diag":
        return v / mass_params['mass_diag_curr'][None, :]  # (N, d)
    else:
        return np.linalg.solve(mass_params['mass_curr'], v.T).T  # (N, d)
