import time
import numpy as np
from typing import Optional
from numpy.typing import NDArray
from scipy.special import logsumexp

from .leapfrog_integration import leapfrog
from .weight_computations import compute_weights_and_ess
from .step_size_adaptation import sample_epsilons, estimate_new_epsilon_mean
from .utils import next_annealing_param


def hamiltonian_snippet(N: int, T: int, step_size: float, mass_diag: NDArray, ESSrmin: float, sample_prior: callable,
                        compute_likelihoods_priors_gradients: callable, adapt_mass: False, verbose: bool = True,
                        seed: Optional[int] = None):
    """Hamiltonian Snippets with Leapfrog integration and step size adaptation.

    Parameters
    ----------
    :param N: Number of particles
    :type N: int
    :param T: Number of integration steps
    :type T: int
    :param step_size: Step size for leapfrog integration
    :type step_size: float
    :param mass_diag: Diagonal of the mass matrix
    :type mass_diag: np.ndarray
    :param ESSrmin: Proportion of `N` that we target as our ESS when finding the next tempering parameter
    :type ESSrmin: float
    :param sample_prior: Function to sample from the prior, should take `N` and `rng` as arguments and return an array
    :type sample_prior: callable
    :param compute_likelihoods_priors_gradients: Function that takes positions of shape `(N, T+1, d)` as input and
                                                 computes the negative log-likelihood, its gradient, the negative
                                                 log prior and its gradient at these positions
    :type compute_likelihoods_priors_gradients: callable
    :param adapt_mass: Whether to adapt the mass matrix diaonal or not
    :type adapt_mass: bool
    :param verbose: Whether to print progress of the algorithm
    :type verbose: bool
    :param seed: Seed for the random number generator
    :type seed: int or None
    """
    assert isinstance(N, int) and N >= 1, "Number of particles must be a positive integer."
    assert isinstance(T, int) and T >= 1, "Number of integration steps must be a positive integer."

    # Set up time-keeping, random number generation, printing, iterations, mass_matrix and more
    start_time = time.time()
    rng = np.random.default_rng(seed=seed if seed is not None else np.random.randint(low=0, high=10000000))
    verboseprint = print if verbose else lambda *a, **kwargs: None
    n = 1

    # Initialize particles, epsilons
    x = sample_prior(N, rng)
    d = x.shape[1]
    v = rng.normal(loc=0, scale=1, size=(N, d)) / np.sqrt(mass_diag)

    # Initial step sizes and mass matrix
    epsilons = sample_epsilons(epsilon_mean=step_size, N=N, rng=rng)
    mass_diag_curr = mass_diag if mass_diag is not None else np.eye(d)

    # Storage
    epsilons_history = [epsilons]
    gammas = [0.0]
    ess_history = [N]
    logLt = 0.0

    while gammas[n-1] < 1.0:
        verboseprint(f"Iteration {n} Gamma {gammas[n-1]: .3f} Avg Step Size: {step_size: .4f}")

        # Construct trajectories
        xnk, vnk, nlps, nlls = leapfrog(x, v, T, epsilons, gammas[n-1], 1/mass_diag, compute_likelihoods_priors_gradients)
        verboseprint("\tTrajectories constructed.")

        # Select next tempering parameter based on target ESS
        gammas.append(next_annealing_param(gamma=gammas[n-1], ESSrmin=ESSrmin, llk=(-nlls[:, 0])))
        verboseprint(f"\tNext gamma selected: {gammas[-1]: .5f}")

        # Estimate new mass matrix diagonal using importance sampling
        if adapt_mass:
            W_mass_est, _, _, _, _ = compute_weights_and_ess(
                vnk, nlps, nlls, 1/mass_diag_curr, 1/mass_diag_curr, gammas[n], gammas[n-1]
            )
            weighted_mean = np.average(xnk.reshape(-1, d), axis=0, weights=W_mass_est.ravel())  # (d, )
            mass_diag_next = 1 / np.average((xnk.reshape(-1, d) - weighted_mean)**2, axis=0, weights=W_mass_est.ravel())
            verboseprint(f"\tNew mass matrix diagonal estimated. Mean {mass_diag_next.mean()}")
        else:
            mass_diag_next = mass_diag_curr

        # Compute weights and ESS
        W_unfolded, logw_unfolded, W_folded, logw_folded, ess = compute_weights_and_ess(
            vnk, nlps, nlls, 1/mass_diag_next, 1/mass_diag_curr, gammas[n], gammas[n-1])
        verboseprint(f"\tWeights Computed. Folded ESS {ess: .3f}")

        # Resample N particles out of N*(T+1) proportionally to unfolded weights
        A = rng.choice(a=N*(T+1), size=N, replace=True, p=W_unfolded.ravel())  # (N, )
        i_indices, k_indices = np.unravel_index(A, (N, T+1))  # (N, ) particles indices, (N, ) trajectory indices
        x = xnk[i_indices, k_indices]  # (N, d) resampled positions
        verboseprint(f"\tParticles resampled. PM {np.sum(k_indices > 0) / N: .3f}")

        # Refresh velocities
        v = rng.normal(loc=0, scale=1, size=(N, d)) / np.sqrt(mass_diag)
        verboseprint("\tVelocities refreshed.")

        # Compute log-normalizing constant estimates
        logLt += logsumexp(logw_folded) - np.log(N)
        verboseprint(f"\tLogLt {logLt}")

        # Step size adaptation
        step_size = estimate_new_epsilon_mean(xnk=xnk, logw=logw_unfolded, epsilons=epsilons, ss=lambda _eps: _eps)
        epsilons = sample_epsilons(epsilon_mean=step_size, N=N, rng=rng)
        verboseprint(f"\tStep size adapted {step_size}")

        # Storage
        epsilons_history.append(epsilons)
        ess_history.append(ess)

        n += 1
    runtime = time.time() - start_time
    return {"logLt": logLt, "gammas": gammas, "runtime": runtime, "epsilons": epsilons_history, "ess": ess_history}
