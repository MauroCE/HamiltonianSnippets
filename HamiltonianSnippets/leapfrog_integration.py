import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def leapfrog(x: NDArray, v: NDArray, T: int, epsilons: NDArray, gamma_curr: float, mass_params: dict,
             compute_likelihoods_priors_gradients: callable) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Leapfrog integration """
    N, d = x.shape
    if len(epsilons.shape) == 1:
        epsilons = epsilons[:, None]
    # inv_mass_diag_curr = inv_mass_diag_curr[None, :]

    # Select momentum update function
    mom_update = lambda gnlp, gnll: epsilons*(gnlp + gamma_curr*gnll)
    if gamma_curr == 0:
        mom_update = lambda gnlp, gnll: epsilons*gnlp

    # Initialize snippets, negative log priors and negative log likelihoods
    xnk = np.full((N, T+1, d), np.nan)
    vnk = np.full((N, T+1, d), np.nan)
    nlps = np.full((N, T+1), np.nan)  # negative log priors
    nlls = np.full((N, T+1), np.nan)  # negative log likelihoods
    xnk[:, 0] = x  # seed positions
    vnk[:, 0] = v  # seed velocities

    # First momentum half-step
    nlps[:, 0], gnlps, nlls[:, 0], gnlls = compute_likelihoods_priors_gradients(x)
    v = v - 0.5*mom_update(gnlp=gnlps, gnll=gnlls)  # *epsilons*(gnlps + gamma_curr*gnlls)  # (N, d)

    # T - 1 position and velocity full steps
    for k in range(T-1):

        # Full position step
        x = x + epsilons*inv_mass_times_v(v, mass_params)  # (v * inv_mass_diag_curr)

        # Full momentum step
        nlps[:, k+1], gnlps, nlls[:, k+1], gnlls = compute_likelihoods_priors_gradients(x)
        v = v - mom_update(gnlp=gnlps, gnll=gnlls)  # epsilons*(gnlps + gamma_curr*gnlls)  # TODO: when gamma=0 and gnlls contain inf then 0*inf=nan

        # Store positions and velocities
        vnk[:, k+1] = v
        xnk[:, k+1] = x

    # Final position half-step
    x = x + epsilons*inv_mass_times_v(v, mass_params)  #(v * inv_mass_diag_curr)

    # Final momentum half-step
    nlps[:, -1], gnlps, nlls[:, -1], gnlls = compute_likelihoods_priors_gradients(x)
    v = v - 0.5*mom_update(gnlp=gnlps, gnll=gnlls)  # epsilons*(gnlps + gamma_curr*gnlls)

    # Store final position and velocity
    xnk[:, -1] = x
    vnk[:, -1] = v

    return xnk, vnk, nlps, nlls


def inv_mass_times_v(v: NDArray, mass_params: dict):
    """Computes M^{-1} v for the various scenarios."""
    if mass_params['matrix_type'] == "diag":
        return v / mass_params['mass_diag_curr'][None, :]  # (N, d)
    else:
        return np.linalg.solve(mass_params['mass_curr'], v.T).T  # (N, d)
