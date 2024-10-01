import numpy as np


def leapfrog(x, v, epsilons):
    """Leapfrog integration """
    pass


def integrate_forward(x, v, epsilons, params, gammas, n):
    """Integrates forward performing Leapfrog integration, with a diagonal mass matrix, meaning that the
    momenta need to be rescaled."""
    N = params['N_particles']
    T = params['T']
    d = params['dim']
    # Setup storage
    trajectories = np.full((N, T+1, 2*d), np.nan)
    trajectories[:, 0] = np.hstack((x, v))
    # Store only the log densities (used for log-weights), not gradients
    nlps = np.full((N, T+1), np.nan)   # Negative log priors
    nlls = np.full((N, T+1), np.nan)   # Negative log-likelihoods
    # First half-momentum step
    nlps[:, 0], gnlps, nlls[:, 0], gnlls = nlp_gnlp_nll_and_gnll(x, params)
    v = v - 0.5*epsilons[:, None]*(gnlps + gammas[n-1]*gnlls)  # (N, d)
    # T-1 full position and momentum steps
    for k in range(T - 1):
        # Full position step with diagonal covariance matrix
        x = x + epsilons[:, None]*(v * params['covariance_diag'])
        # Full momentum step (compute nlls and gnlls)
        nlps[:, k+1], gnlps, nlls[:, k+1], gnlls = nlp_gnlp_nll_and_gnll(x, params)
        v = v - epsilons[:, None]*(gnlps + gammas[n-1]*gnlls)  # (N, 61)
        # Store trajectory
        trajectories[:, k+1] = np.hstack((x, v))
    # Final position half-step
    x = x + epsilons[:, None]*(v * params['covariance_diag'])
    # Final momentum half-step
    nlps[:, -1], gnlps, nlls[:, -1], gnlls = nlp_gnlp_nll_and_gnll(x, params)
    v = v - 0.5*epsilons[:, None]*(gnlps + gammas[n-1]*gnlls)  # (N, 61)
    # Store trajectories
    trajectories[:, -1] = np.hstack((x, v))
    return trajectories, nlps, nlls
