import numpy as np


def leapfrog(x, v, T, epsilons, gamma_curr, inv_mass_diag_curr, compute_likelihoods_priors_gradients):
    """Leapfrog integration """
    N, d = x.shape
    if len(epsilons.shape) == 1:
        epsilons = epsilons[:, None]
    inv_mass_diag_curr = inv_mass_diag_curr[None, :]

    # Initialize snippets, negative log priors and negative log likelihoods
    xnk = np.full((N, T+1, d), np.nan)
    vnk = np.full((N, T+1, d), np.nan)
    nlps = np.full((N, T+1), np.nan)  # negative log priors
    nlls = np.full((N, T+1), np.nan)  # negative log likelihoods
    xnk[:, 0] = x  # seed positions
    vnk[:, 0] = v  # seed velocities

    # First momentum half-step
    nlps[:, 0], gnlps, nlls[:, 0], gnlls = compute_likelihoods_priors_gradients(x)
    v = v - 0.5*epsilons*(gnlps + gamma_curr*gnlls)  # (N, d)

    # T - 1 position and velocity full steps
    for k in range(T-1):

        # Full position step
        x = x + epsilons*(v * inv_mass_diag_curr)

        # Full momentum step
        nlps[:, k+1], gnlps, nlls[:, k+1], gnlls = compute_likelihoods_priors_gradients(x)
        v = v - epsilons*(gnlps + gamma_curr*gnlls)

        # Store positions and velocities
        vnk[:, k+1] = v
        xnk[:, k+1] = x

    # Final position half-step
    x = x + epsilons*(v * inv_mass_diag_curr)

    # Final momentum half-step
    nlps[:, -1], gnlps, nlls[:, -1], gnlls = compute_likelihoods_priors_gradients(x)
    v = v - 0.5*epsilons*(gnlps + gamma_curr*gnlls)

    # Store final position and velocity
    xnk[:, -1] = x
    vnk[:, -1] = v

    return xnk, vnk, nlps, nlls
