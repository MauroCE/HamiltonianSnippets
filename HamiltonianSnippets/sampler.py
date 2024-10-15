import time
import numpy as np
from typing import Optional
from numpy.typing import NDArray
from scipy.special import logsumexp
from copy import deepcopy

from .leapfrog_integration import leapfrog
from .weight_computations import compute_weights
from .step_size_adaptation import sample_epsilons, estimate_with_cond_variance
from .utils import next_annealing_param
from .num_leapfrog_steps_adaptation import adapt_num_leapfrog_steps_contractivity


def hamiltonian_snippet(N: int, T: int, mass_diag: NDArray, ESSrmin: float, sample_prior: callable,
                        compute_likelihoods_priors_gradients: callable, epsilon_params: dict,
                        act_on_overflow: bool = False, adapt_step_size: bool = False,
                        adapt_n_leapfrog_steps: bool = False, skip_overflown: bool = False, adapt_mass: bool = False,
                        plot_contractivity: bool = False, T_max: int = 100, T_min: int = 5,
                        max_tries_find_coupling: int = 100,
                        verbose: bool = True, seed: Optional[int] = None):
    """Hamiltonian Snippets with Leapfrog integration and step size adaptation.

    Parameters
    ----------
    :param N: Number of particles
    :type N: int
    :param T: Number of integration steps
    :type T: int
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
    :param epsilon_params: Parameters for the distribution of the epsilons. Should be a dictionary containing
                           'distribution' which should be one of `['inv_gauss']` and parameters for them
    :type epsilon_params: dict
    :param act_on_overflow: Whether to use contingency measures in epsilon_params['on_overflow'] when overflow found
    :type act_on_overflow: bool
    :param adapt_step_size: Whether to adapt the leapfrog step size
    :type adapt_step_size: bool
    :param adapt_n_leapfrog_steps: Whether to adapt the number of leapfrog steps
    :type adapt_n_leapfrog_steps: bool
    :param skip_overflown: Whether to skip overflown trajectories when estimating epsilon
    :type skip_overflown: bool
    :param adapt_mass: Whether to adapt the mass matrix diagonal or not
    :type adapt_mass: bool
    :param plot_contractivity: Whether to plot the contractivity at each step
    :type plot_contractivity: bool
    :param T_max: Maximum budget for the number of integration steps
    :type T_max: int
    :param T_min: Minimum budget for the number of integration steps
    :type T_min: int
    :param max_tries_find_coupling: Number of maximum tries used to try to find a coupling
    :type max_tries_find_coupling: int
    :param verbose: Whether to print progress of the algorithm
    :type verbose: bool
    :param seed: Seed for the random number generator
    :type seed: int or None
    """
    assert isinstance(N, int) and N >= 1, "Number of particles must be a positive integer."
    assert isinstance(T, int) and T >= 1, "Number of integration steps must be a positive integer."
    assert T_max > T_min, "Maximum number of integration steps must be larger than minimum number."
    assert max_tries_find_coupling > 1, "Maximum number of tries to find a coupling must be >= 1."

    # Set up time-keeping, random number generation, printing, iterations, mass_matrix and more
    start_time = time.time()
    rng = np.random.default_rng(seed=seed if seed is not None else np.random.randint(low=0, high=10000000))
    verboseprint = print if verbose else lambda *a, **kwargs: None
    n = 1

    # Initialize particles, epsilons
    x = sample_prior(N, rng)
    d = x.shape[1]
    mass_diag_curr = mass_diag if mass_diag is not None else np.eye(d)
    v = rng.normal(loc=0, scale=1, size=(N, d)) * np.sqrt(mass_diag_curr)

    # Initial step sizes and mass matrix
    epsilons = sample_epsilons(eps_params=epsilon_params, N=N, rng=rng)

    # Storage
    epsilon_history = [epsilons]
    epsilon_params_history = [{key: value for key, value in epsilon_params.items() if key != "params_to_estimate"}]  # parameters for the epsilon distribution
    gammas = [0.0]
    ess_history = [N]
    logLt = 0.0
    T_history = [T]
    coupling_success_history = []

    while gammas[n-1] < 1.0:
        verboseprint(f"Iteration {n} Gamma {gammas[n-1]: .5f} Epsilon {epsilon_params['to_print'].capitalize()}: {epsilon_params[epsilon_params['to_print']]}")

        # Construct trajectories
        xnk, vnk, nlps, nlls = leapfrog(x, v, T, epsilons, gammas[n-1], 1/mass_diag_curr, compute_likelihoods_priors_gradients)
        verboseprint("\tTrajectories constructed.")

        # Check if there is any inf due to overflow error
        if not (np.all(np.isfinite(xnk)) and np.all(np.isfinite(vnk)) and
                np.all(np.isfinite(nlps)) and np.all(np.isfinite(nlls))):
            overflow_mask = np.any(~np.isfinite(xnk), axis=2) | np.any(~np.isfinite(vnk), axis=2) | ~np.isfinite(nlps) | ~np.isfinite(nlls)  # (N, T+1)
            verboseprint(f"\tOverflow Detected. Trajectories affected: {overflow_mask.any(axis=1).sum()}")
            # When there is overflow do something
            if act_on_overflow:
                dict_update = epsilon_params['on_overflow'](epsilon_params)
                for key, value in dict_update.items():
                    verboseprint(f"\tOn Overflow changed Epsilon {key} to {value}.")
                epsilon_params.update(dict_update)
        else:
            overflow_mask = np.zeros((N, T+1), dtype=bool)

        # Select next tempering parameter based on target ESS
        gammas.append(next_annealing_param(gamma=gammas[n-1], ESSrmin=ESSrmin, llk=(-nlls[:, 0])))
        verboseprint(f"\tNext gamma selected: {gammas[-1]: .5f}")

        # Estimate new mass matrix diagonal using importance sampling
        if adapt_mass:
            W_mass_est, _, _, _, _ = compute_weights(
                vnk, nlps, nlls, 1/mass_diag_curr, 1/mass_diag_curr, gammas[n], gammas[n-1],
                overflow_mask=overflow_mask
            )
            weighted_mean = np.average(xnk.reshape(-1, d)[~overflow_mask.ravel()], axis=0, weights=W_mass_est.ravel()[~overflow_mask.ravel()])  # (d, )
            mass_diag_next = 1 / np.average((xnk.reshape(-1, d)[~overflow_mask.ravel()] - weighted_mean)**2, axis=0, weights=W_mass_est.ravel()[~overflow_mask.ravel()])
            verboseprint(f"\tNew mass matrix diagonal estimated. Mean {mass_diag_next.mean()}")
        else:
            mass_diag_next = mass_diag_curr

        # Compute weights and ESS
        W_unfolded, logw_unfolded, W_folded, logw_folded, logw_criterion = compute_weights(
            vnk, nlps, nlls, 1/mass_diag_next, 1/mass_diag_curr, gammas[n], gammas[n-1], overflow_mask=overflow_mask)
        ess = 1 / np.sum(W_folded**2)  # folded ESS
        verboseprint(f"\tWeights Computed. Folded ESS {ess: .3f}")

        # Set the new cov matrix to the old one
        mass_diag_curr = mass_diag_next

        # Resample N particles out of N*(T+1) proportionally to unfolded weights
        A = rng.choice(a=N*(T+1), size=N, replace=True, p=W_unfolded.ravel())  # (N, )
        i_indices, k_indices = np.unravel_index(A, (N, T+1))  # (N, ) particles indices, (N, ) trajectory indices
        x = xnk[i_indices, k_indices]  # (N, d) resampled positions
        verboseprint(f"\tParticles resampled. PM {np.sum(k_indices > 0) / N: .3f}")

        # Refresh velocities
        v = rng.normal(loc=0, scale=1, size=(N, d)) * np.sqrt(mass_diag_curr)
        verboseprint("\tVelocities refreshed.")

        # Compute log-normalizing constant estimates
        logLt += logsumexp(logw_folded) - np.log(N)
        verboseprint(f"\tLogLt {logLt}")

        # Step size adaptation
        if adapt_step_size:
            xnk[overflow_mask] = 0.0
            epsilon_params.update(estimate_with_cond_variance(
                xnk=xnk, logw=logw_criterion, epsilons=epsilons, ss_dict=epsilon_params['params_to_estimate'],
                skip_overflown=skip_overflown, overflow_mask=overflow_mask.any(axis=1)
            ))
        new_epsilons = sample_epsilons(eps_params=epsilon_params, N=N, rng=rng)
        verboseprint(f"\tEpsilon {epsilon_params['to_print'].capitalize()} {epsilon_params[epsilon_params['to_print']]}")

        # T adaptation
        if adapt_n_leapfrog_steps:
            T, coupling_found = adapt_num_leapfrog_steps_contractivity(
                xnk=xnk, vnk=vnk, epsilons=epsilons, nlps=nlps, nlls=nlls, T=T, gamma=gammas[n-1],
                inv_mass_diag=1/mass_diag_curr, compute_likelihoods_priors_gradients=compute_likelihoods_priors_gradients,
                plot_contractivity=plot_contractivity, max_tries=max_tries_find_coupling, T_max=T_max, T_min=T_min,
                rng=rng)
            coupling_success_history.append(coupling_found)
        verboseprint(f"\tT: {T}")
        epsilons = new_epsilons  # do it after, to avoid scoping/deepcopy issues

        # Storage
        epsilon_history.append(epsilons)
        epsilon_params_history.append({key: value for key, value in epsilon_params.items() if key != "params_to_estimate"})
        ess_history.append(ess)
        T_history.append(T)

        n += 1
    runtime = time.time() - start_time
    return {"logLt": logLt, "gammas": gammas, "runtime": runtime, "epsilons": epsilon_history, "ess": ess_history,
            'epsilon_params_history': epsilon_params_history, 'T_history': T_history,
            'coupling_success_history': coupling_success_history}
