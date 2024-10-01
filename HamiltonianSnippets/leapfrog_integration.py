import numpy as np


def leapfrog(x, v, epsilons, gamma_curr, inv_mass_diag, compute_likelihoods_priors_gradients):
    """Leapfrog integration """
    N, Tplus1, d = x.shape
    if len(epsilons.shape) == 1:
        epsilons = epsilons[:, None]
    inv_mass_diag = inv_mass_diag[None, :]

    # Initialize snippets, negative log priors and negative log likelihoods
    znk = np.full((N, Tplus1, 2*d), np.nan)  # snippets
    nlps = np.full((N, Tplus1), np.nan)  # negative log priors
    nlls = np.full((N, Tplus1), np.nan)  # negative log likelihoods
    znk[:, 0] = np.hstack((x, v))  # Fill with seed particles

    # First momentum half-step
    nlps[:, 0], gnlps, nlls[:, 0], gnlls = compute_likelihoods_priors_gradients(x)
    v = v - 0.5*epsilons*(gnlps + gamma_curr*gnlls)  # (N, d)

    # T - 1 position and velocity full steps
    for k in range(Tplus1 - 2):

        # Full position step
        x = x + epsilons*(v * inv_mass_diag)

        # Full momentum step
        nlps[:, k+1], gnlps, nlls[:, k+1], gnlls = compute_likelihoods_priors_gradients(x)
        v = v - epsilons*(gnlps + gamma_curr*gnlls)

        # Store trajectories
        znk[:, k+1] = np.hstack((x, v))

    # Final position half-step
    x = x + epsilons*(v * inv_mass_diag)

    # Final momentum half-step
    nlps[:, -1], gnlps, nlls[:, -1], gnlls = compute_likelihoods_priors_gradients(x)
    v = v - 0.5*epsilons*(gnlps + gamma_curr*gnlls)

    # Store trajectories
    znk[:, -1] = np.hstack((x, v))
    return znk, nlps, nlls
