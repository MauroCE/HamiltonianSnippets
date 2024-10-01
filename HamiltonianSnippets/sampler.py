import time
import numpy as np
from typing import Optional
from numpy.typing import NDArray
from scipy.special import logsumexp
from leapfrog_integration import leapfrog

from utils import next_annealing_param


def hamiltonian_snippet(T: int, mass_matrix: NDArray, N: int, ESSrmin: float, sample_prior: callable, verbose: bool = True, seed: Optional[int] = None):
    """Some"""
    # Set up time-keeping, random number generation, printing, iterations, mass_matrix and more
    start_time = time.time()
    rng = np.random.default_rng(seed=seed if seed is not None else np.random.randint(low=0, high=10000000))
    verboseprint = print if verbose else lambda *a, **kwargs: None
    n = 1
    M_chol = np.linalg.cholesky(mass_matrix)  # (d, d) lower-triangular

    # Initialize particles and epsilons
    x = sample_prior(N, rng)
    d = x.shape[1]
    v = (M_chol @ rng.normal(loc=0, scale=1, size=(N, d)).T).T

    # Storage
    gammas = [0.0]
    logLt = 0.0

    while gammas[n-1] < 1.0:
        verboseprint(f"Iteration {n} Gamma {gammas[n-1]}")

        # Construct trajectories
        znk, nlps, nlls = leapfrog(x, v, epsilons, gammas[n-1], M_inv, compute_likelihoods_priors_gradients)

        # Select next tempering parameter based on target ESS
        gammas.append(next_annealing_param(gamma=gammas[n-1], ESSrmin=ESSrmin, llk=(-nlls[:, 0])))

        # Compute unfolded weights and ESS
        W_unfolded, logw_unfolded, W_folded, logw_folded, ess = compute_weights()

        # Resample N particles out of N*(T+1) proportionally to unfolded weights
        A = rng.choice(a=N*(T+1), size=N, replace=True, p=W_unfolded.ravel())  # (N, )
        i_indices, k_indices = np.unravel_index(A, (N, T+1))  # (N, ) particles indices, (N, ) trajectory indices
        x = znk[i_indices, k_indices, :d]    # (N, d) resampled positions

        # Refresh velocities
        v = (M_chol @ rng.normal(loc=0, scale=1, size=(N, d)).T).T

        # Compute log-normalizing constant estimates
        logLt += logsumexp(logw_folded) - np.log(N)

        # Step size adaptation

        # Storage
        pass
    pass
