import time
import numpy as np
from typing import Optional
from scipy.special import logsumexp
from numpy.typing import NDArray
from copy import deepcopy

from .leapfrog_integration import leapfrog
from .weight_computations import compute_weights
from .step_size_adaptation import sample_epsilons, estimate_with_cond_variance
from .mass_matrix_adaptation import update_mass_matrix
from .utils import next_annealing_param
from .num_leapfrog_steps_adaptation import adapt_num_leapfrog_steps_contractivity


def hamiltonian_snippet(N: int, T: int, ESSrmin: float, sample_prior: callable,
                        compute_likelihoods_priors_gradients: callable, epsilon_params: dict,
                        mass_params: dict,
                        act_on_overflow: bool = False, adapt_step_size: bool = False,
                        adapt_n_leapfrog_steps: bool = False, skip_overflown: bool = False,
                        plot_contractivity: bool = False, plot_Q_criterion: bool = False, T_max: int = 100,
                        T_min: int = 5,
                        max_contractivity: int = 3,
                        max_tries_find_coupling: int = 100,
                        verbose: bool = True, seed: Optional[int] = None) -> dict:
    """Hamiltonian Snippets with Leapfrog integration and step size adaptation.

    Parameters
    ----------
    :param N: Number of particles
    :type N: int
    :param T: Number of integration steps
    :type T: int
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
    :param mass_params: Parameters for mass matrix
    :type mass_params: dict
    :param plot_contractivity: Whether to plot the contractivity at each step
    :type plot_contractivity: bool
    :param plot_Q_criterion: Whether to plot the Q criterion at each iteration
    :type plot_Q_criterion: bool
    :param T_max: Maximum budget for the number of integration steps
    :type T_max: int
    :param T_min: Minimum budget for the number of integration steps
    :type T_min: int
    :param max_contractivity: Maximum contractivity. Larger values are clipped to this
    :type max_contractivity: int
    :param max_tries_find_coupling: Number of maximum tries used to try to find a coupling
    :type max_tries_find_coupling: int
    :param verbose: Whether to print progress of the algorithm
    :type verbose: bool
    :param seed: Seed for the random number generator
    :type seed: int or None
    :return: A dictionary with the results of the Hamiltonian snippet
    :rtype: dict
    """
    assert isinstance(N, int) and N >= 1, "Number of particles must be a positive integer."
    assert isinstance(T, int) and T >= 1, "Number of integration steps must be a positive integer."
    assert T_max >= T_min, "Maximum number of integration steps must be larger than minimum number."
    assert max_tries_find_coupling > 1, "Maximum number of tries to find a coupling must be >= 1."

    # Set up time-keeping, random number generation, printing, iterations, mass_matrix and more
    start_time = time.time()
    rng = np.random.default_rng(seed=seed if seed is not None else np.random.randint(low=0, high=10000000))
    verboseprint = print if verbose else lambda *a, **kwargs: None
    mass_params = process_mass_params(mass_params)
    n = 1

    # Initialize particles, epsilons
    x = sample_prior(N, rng)
    d = x.shape[1]
    # mass_diag_curr = mass_diag if mass_diag is not None else np.eye(d)
    v = sample_velocities(mass_params, N, d, rng)  # (N, d)
    # v = transform_v(rng.normal(loc=0, scale=1, size=(N, d)), mass_params, 0.0)  # rng.normal(loc=0, scale=1, size=(N, d)) * np.sqrt(mass_diag_curr)

    # Initial step sizes and mass matrix
    epsilons = sample_epsilons(eps_params=epsilon_params, N=N, rng=rng)

    # Storage
    epsilon_history = [epsilons]
    epsilon_params_history = [{key: value for key, value in epsilon_params.items() if key not in ["params_to_estimate", "on_overflow"]}]  # parameters for the epsilon distribution
    gammas = [0.0]
    ess_history = [N]
    logLt = 0.0
    T_history = [T]
    coupling_success_history = []

    while gammas[n-1] < 1.0:
        verboseprint(f"Iteration {n} Gamma {gammas[n-1]: .5f} Epsilon {epsilon_params['to_print'].capitalize()}: {epsilon_params[epsilon_params['to_print']]}")

        # Construct trajectories
        xnk, vnk, nlps, nlls = leapfrog(x, v, T, epsilons, gammas[n-1], mass_params, compute_likelihoods_priors_gradients)
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
        mass_params = update_mass_matrix(mass_params=mass_params, xnk=xnk, vnk=vnk, nlps=nlps, nlls=nlls, gammas=gammas, n=n, overflow_mask=overflow_mask)

        # Compute weights and ESS
        W_unfolded, logw_unfolded, W_folded, logw_folded, logw_criterion = compute_weights(
            vnk=vnk, nlps=nlps, nlls=nlls, mass_params=mass_params, gamma_next=gammas[n], gamma_curr=gammas[n-1], overflow_mask=overflow_mask
        )
        ess = 1 / np.sum(W_folded**2)  # folded ESS
        verboseprint(f"\tWeights Computed. Folded ESS {ess: .3f}")

        # Resample N particles out of N*(T+1) proportionally to unfolded weights
        A = rng.choice(a=N*(T+1), size=N, replace=True, p=W_unfolded.ravel())  # (N, )
        i_indices, k_indices = np.unravel_index(A, (N, T+1))  # (N, ) particles indices, (N, ) trajectory indices
        x = xnk[i_indices, k_indices]  # (N, d) resampled positions
        verboseprint(f"\tParticles resampled. PM {np.sum(k_indices > 0) / N: .3f}")

        # Compute log-normalizing constant estimates
        logLt += logsumexp(logw_folded) - np.log(N)
        verboseprint(f"\tLogLt {logLt}")

        # Step size adaptation
        # TODO: make sure xnk[overflow_mask] = 0.0 is not fucking up T adaptation
        if adapt_step_size:
            xnk_for_eps_adaptation = deepcopy(xnk)
            xnk_for_eps_adaptation[overflow_mask] = 0.0
            epsilon_params.update(estimate_with_cond_variance(
                xnk=xnk_for_eps_adaptation, logw_criterion=logw_criterion, epsilons=epsilons, epsilon_params=epsilon_params,
                skip_overflown=skip_overflown, overflow_mask=overflow_mask.any(axis=1), plot_Q_criterion=plot_Q_criterion
            ))
        new_epsilons = sample_epsilons(eps_params=epsilon_params, N=N, rng=rng)
        verboseprint(f"\tEpsilon {epsilon_params['to_print'].capitalize()} {epsilon_params[epsilon_params['to_print']]}")

        # T adaptation
        if adapt_n_leapfrog_steps:
            T, coupling_found = adapt_num_leapfrog_steps_contractivity(
                xnk=xnk, vnk=vnk, epsilons=epsilons, nlps=nlps, nlls=nlls, T=T, gamma=gammas[n-1],
                mass_params=mass_params, compute_likelihoods_priors_gradients=compute_likelihoods_priors_gradients,
                plot_contractivity=plot_contractivity, max_tries=max_tries_find_coupling, T_max=T_max, T_min=T_min,
                max_contractivity=max_contractivity,
                rng=rng)
            coupling_success_history.append(coupling_found)
        verboseprint(f"\tT: {T}")
        epsilons = new_epsilons  # do it after, to avoid scoping/deepcopy issues

        # Set the new cov matrix to the old one
        mass_params = curr_mass_becomes_next_mass(mass_params)  # mass_diag_curr = mass_diag_next

        # Refresh velocities
        v = sample_velocities(mass_params, N, d, rng)  # rng.normal(loc=0, scale=1, size=(N, d)) * np.sqrt(mass_diag_curr)
        verboseprint("\tVelocities refreshed.")

        # Storage
        epsilon_history.append(epsilons)
        epsilon_params_history.append({key: value for key, value in epsilon_params.items() if key not in ["params_to_estimate", "on_overflow"]})
        ess_history.append(ess)
        T_history.append(T)

        n += 1
    runtime = time.time() - start_time
    return {"logLt": logLt, "gammas": gammas, "runtime": runtime, "epsilons": epsilon_history, "ess": ess_history,
            'epsilon_params_history': epsilon_params_history, 'T_history': T_history,
            'coupling_success_history': coupling_success_history}


def curr_mass_becomes_next_mass(mass_params: dict) -> dict:
    """We set curr <- next so that we can refresh the velocities easily using the same function.

    Parameters
    ----------
    :param mass_params: Current mass matrix parameters
    :type mass_params: dict
    :return: Updated mass matrix parameters where everything with 'curr' is overwritten by 'next'
    :rtype: dict
    """
    term = '' if mass_params['matrix_type'] == 'full' else 'diag_'
    mass_params[f'mass_{term}curr'] = mass_params[f'mass_{term}next']
    mass_params[f'chol_mass_{term}curr'] = mass_params[f'chol_mass_{term}next']
    mass_params['log_det_mass_curr'] = mass_params['log_det_mass_next']
    return mass_params


def process_mass_params(mass_params: dict) -> dict:
    """Pre-processes mass params by filling-in the current mass matrix and the next mass matrix, the latter of which
    is set to be identical to the current one. It will be overwritten later, but it is necessary for mass matrix
    adaptation.

    Parameters
    ----------
    :param mass_params: Mass matrix parameters
    :type mass_params: dict
    :return: Post-processed mass matrix parameters
    :rtype: dict
    """
    assert "strategy" in mass_params, "Mass Matrix parameters must contain 'strategy'."
    assert "matrix_type" in mass_params, "Mass Matrix parameters must contain `matrix_type`."
    assert mass_params['strategy'] in {'fixed', 'schedule', 'adaptive'}, "Mass Matrix strategy must be one of `fixed, `schedule` or `adaptive`."
    assert mass_params['matrix_type'] in {'diag', 'full'}, "Mass matrix type must be one of `diag` or `full`."
    assert (mass_params['matrix_type'] == "full" and len(mass_params['mass'].shape) == 2) or (mass_params['matrix_type'] == "diag" and len(mass_params['mass'].shape) == 1), "When matrix_type=full, then one must provide mass to be a matrix, when matrix_type=diag, one must provide the vector corresponding to its diagonal"
    if mass_params['strategy'] == 'adaptive' and mass_params['matrix_type'] == 'full':
        raise NotImplementedError("Mass Matrix adaptation with a full mass matrix is not implemented.")
    if mass_params["strategy"] == "schedule" and "schedule_func" not in mass_params:
        raise ValueError("When using a full mass matrix schedule, one must provide a schedule function.")
    term = '' if mass_params['matrix_type'] == 'full' else 'diag_'

    # Set the current mass matrix, cholesky, log det, etc. (current meaning at gamma=0.0)
    mass_params[f'mass_{term}curr'] = mass_params['mass'] if mass_params['strategy'] != "schedule" else mass_params['schedule_func'](0.0)
    mass_params[f'chol_mass_{term}curr'] = np.linalg.cholesky(mass_params['mass']) if mass_params['matrix_type'] == "full" else np.sqrt(mass_params['mass'])
    mass_params['log_det_mass_curr'] = np.linalg.slogdet(mass_params['mass']).logabsdet if mass_params['matrix_type'] == "full" else np.sum(np.log(mass_params['mass']))

    if mass_params['strategy'] == "fixed" or mass_params['strategy'] == "adaptive":
        # Set next = curr. For adaptive, we do it so that we can compute the weights used in the mass matrix adaptation, since
        # those will require
        mass_params[f'mass_{term}next'] = mass_params[f'mass_{term}curr']
        mass_params[f'chol_mass_{term}next'] = mass_params[f'chol_mass_{term}curr']
        mass_params['log_det_mass_next'] = mass_params['log_det_mass_curr']
    return mass_params


def sample_velocities(mass_params: dict, N: int, d: int, rng: np.random.Generator) -> NDArray:
    """Given samples v of shape (N, d) sampled from a standard normal, we use the mass params
    to sample them from N(0, M).

    Parameters
    ----------
    :param mass_params: Mass matrix parameters
    :type mass_params: dict
    :param N: Number of velocities to sample, should be equal to the number of particles
    :type N: int
    :param d: Dimensionality of the velocities
    :type d: int
    :param rng: Random number generator, used for reproducibility
    :type rng: np.random.generator
    :return: Sampled velocities of shape (N, d)
    :rtype: np.ndarray
    """
    v = rng.normal(loc=0, scale=1, size=(N, d))
    match mass_params['strategy'], mass_params['matrix_type']:

        case ("fixed" | "schedule" | "adaptive"), "diag":
            return mass_params['chol_mass_diag_curr'] * v  # (N, d)

        case ("fixed" | "schedule"), "full":
            return v.dot(mass_params['chol_mass_curr'].T)  # (N, d)
