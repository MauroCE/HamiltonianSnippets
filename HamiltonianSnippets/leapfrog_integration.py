import numpy as np


def leapfrog(x, v, epsilons, gamma_curr, inv_mass_diag_curr, compute_likelihoods_priors_gradients):
    """Leapfrog integration """
    N, Tplus1, d = x.shape
    if len(epsilons.shape) == 1:
        epsilons = epsilons[:, None]
    inv_mass_diag_curr = inv_mass_diag_curr[None, :]

    # Initialize snippets, negative log priors and negative log likelihoods
    xnk = np.full((N, Tplus1, d), np.nan)
    vnk = np.full((N, Tplus1, d), np.nan)
    nlps = np.full((N, Tplus1), np.nan)  # negative log priors
    nlls = np.full((N, Tplus1), np.nan)  # negative log likelihoods
    xnk[:, 0] = x  # seed positions
    vnk[:, 0] = v  # seed velocities

    # First momentum half-step
    nlps[:, 0], gnlps, nlls[:, 0], gnlls = compute_likelihoods_priors_gradients(xnk[:, 0])
    vnk[:, 1] = vnk[:, 0] - 0.5*epsilons*(gnlps + gamma_curr*gnlls)  # (N, d)

    # T - 1 position and velocity full steps
    for k in range(Tplus1 - 2):  # k final is T-2 so k+1 T-1 and array has size T+1 with 0-index final index T

        # Full position step
        xnk[:, k+1] = xnk[:, k] + epsilons*(vnk[:, k+1] * inv_mass_diag_curr)

        # Full momentum step
        nlps[:, k+1], gnlps, nlls[:, k+1], gnlls = compute_likelihoods_priors_gradients(xnk[:, k+1])
        vnk[:, k+1] = vnk[:, k+1] - epsilons*(gnlps + gamma_curr*gnlls)

    # Final position half-step
    xnk[:, -1] = xnk[:, -2] + epsilons*(vnk[:, -2] * inv_mass_diag_curr)

    # Final momentum half-step
    nlps[:, -1], gnlps, nlls[:, -1], gnlls = compute_likelihoods_priors_gradients(xnk[:, -1])
    vnk[:, -1] = vnk[:, -2] - 0.5*epsilons*(gnlps + gamma_curr*gnlls)

    return xnk, vnk, nlps, nlls
