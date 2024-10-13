"""
The aim of this script is to be able to generate figures describing our 'tracking' optimization
procedure. I want to plot our v(z, epsilon) random function, our Inverse Gaussian proposal
the convolution of those and finally the newly chosen Inverse Gaussian.
"""
import os
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
import time
from scipy.special import logsumexp
from scipy.stats import invgauss
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
rc('font', **{'family': 'STIXGeneral'})


def generate_nlp_gnlp_nll_and_gnll_function(_y, _Z, _scales):
    """Computes negative log likelihood and its gradient manually.

    nll(x) = sum_{i=1}^{n_d} log(1 + exp(-y_i x^top z_i))

    gnll(x) = sum_{i=1}^{n_d} frac{exp(-y_i x^top z_i)}{1 + exp(-y_i x^top z_i)} y_i z_i
    """
    def nlp_gnlp_nll_and_gnll(x):
        # Negative log prior
        nlp = 0.5*61*np.log(2*np.pi) + 0.5*np.log(400.*(25.0**60)) + 0.5*np.sum((x / _scales)**2, axis=1)
        gnlp = x / (_scales**2)
        # Here I use D for the number of data points (n_d)
        logE = (-_y[None, :] * x.dot(_Z.T)).T  # (N, n_d)
        laeE = np.logaddexp(0.0, logE)  # (n_d, N)
        gnll = - np.einsum('DN, D, Dp -> Np', np.exp(logE - laeE), _y, _Z)  # (N, 61)
        return nlp, gnlp, np.sum(laeE, axis=0), gnll  # (N, ) and (N, 61)
    return nlp_gnlp_nll_and_gnll


def generate_sample_prior_function(_scales):
    """Samples n particles from the prior."""
    return lambda n, rng: _scales * rng.normal(loc=0.0, scale=1.0, size=(n, 61))


def sample_epsilons(eps_params: dict, N: int, rng: np.random.Generator) -> NDArray:
    """Samples epsilons according to the correct distribution."""
    assert 'distribution' in eps_params, "Epsilon parameters must contain a key 'distribution'."
    match eps_params['distribution']:

        case 'inv_gauss':
            assert eps_params['skewness'] > 0, "Skewness must be strictly positive for an inverse gaussian."
            assert eps_params['mean'] > 0, "Mean must be strictly positive for an inverse gaussian."
            lambda_param = 9*eps_params['mean'] / eps_params['skewness']**2  # since skewness = 3*sqrt(mu/lambda)
            return invgauss.rvs(mu=eps_params['mean']/lambda_param, loc=0, scale=lambda_param, size=N, random_state=rng)

        case 'discrete_uniform':
            assert len(eps_params['values']) > 0, "When using a discrete uniform distribution, there should be at least one value, one for each group."
            assert np.all(np.array(eps_params['values']) > 0), "Epsilon values must be positive for all groups when using a uniform discrete distribution."
            return rng.choice(a=eps_params['values'], size=N, replace=True)


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
        v = v - epsilons*(gnlps + gamma_curr*gnlls)  # TODO: when gamma=0 and gnlls contain inf then 0*inf=nan

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


def ess_from_log_weights(logw: np.ndarray) -> float:
    """ESS (Effective sample size) computed from log-weights, this function has been copied from Nicholas Chopin
    `particles` package.

    Parameters
    ----------
    :param logw: Log weights of shape `(N, )`
    :type logw: np.ndarray
    :return: ESS of weights `w=np.exp(logw)` i.e. the quantity `np.sum(w**2) / np.sum(w)**2`
    :rtype: float
    """
    w = np.exp(logw - logw.max())
    return (w.sum()) ** 2 / np.sum(w ** 2)


def next_annealing_param(gamma: float, ESSrmin: float, llk) -> float:
    """Find next annealing exponent by solving `ESS(exp(logw)) = alpha * N`.

    Parameters
    ----------
    :param gamma: Current tempering parameter, typically denoted $\\gamma_{n-1}$
    :type gamma: float
    :param ESSrmin: Target ESS proportion with respect to `N`, e.g. `0.8` means we target `0.8N` as our ESS
    :type ESSrmin: float
    :param llk: Log likelihoods, has shape `(N, )`
    :type llk: np.ndarray
    :return: New tempering parameter
    :rtype: float
    """
    N = llk.shape[0]

    def f(e):
        ess = ess_from_log_weights(e * llk) if e > 0.0 else N  # avoid 0 x inf issue when e==0
        return ess - ESSrmin * N
    if f(1. - gamma) < 0.:
        return gamma + brentq(f, 0.0, 1.0 - gamma)
    else:
        return 1.0


def compute_weights(
        vnk: NDArray,
        nlps: NDArray,
        nlls: NDArray,
        inv_mass_diag_next: NDArray,
        inv_mass_diag_curr: NDArray,
        gamma_next: float,
        gamma_curr: float,
        overflow_mask: NDArray,
        ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Computes unfolded and folded weights.

    Parameters
    ----------
    :param vnk: Velocities constructed with $\\psi_{n-1}$, has shape `(N, T+1, d)` where `d` is the velocity dimension
    :type vnk: numpy.ndarray
    :param nlps: Negative log prior density on the snippets, has shape `(N, T+1)`
    :type nlps: numpy.ndarray
    :param nlls: Negative log likelihood on the snippets, has shape `(N, T+1)`
    :type nlls: numpy.ndarray
    :param inv_mass_diag_next: Diagonal of the inverse mass matrix at time `n`
    :type inv_mass_diag_next: numpy.ndarray
    :param inv_mass_diag_curr: Diagonal of the inverse mass matrix at time `n-1`
    :type inv_mass_diag_curr: numpy.ndarray
    :param gamma_next: Tempering parameter $\\gamma_n$ for the next distribution, will be used in the numerator
    :type gamma_next: float
    :param gamma_curr: Tempering parameter $\\gamma_{n-1}$ for the current distribution, will be used in the denominator
    :type gamma_curr: float
    :param overflow_mask: Mask indicating where there has been an overflow either in xnk or vnk. There we do not
                          compute a weight, it will be set to zero. Will have shape (N, T+1).
    :type overflow_mask: numpy.ndarray
    :return: Tuple containing `(W_unfolded, logw_unfolded, W_folded, logw_unfolded)`
    :rtype: tuple(numpy.array, numpy.array, numpy.array, numpy.array)
    """
    assert len(vnk.shape) == 3, "Velocities must be a 3-dimensional array."
    assert len(nlps.shape) == 2, "Negative log priors must be a 2-dimensional array."
    assert len(nlls.shape) == 2, "Negative log likelihoods must be a 2-dimensional array."
    N, Tplus1, d = vnk.shape

    # ofm = overflow_mask.ravel()  # (N*(T+1), ) boolean mask, True if corresponding xnk or vnk is inf due to overflow
    # ofm_seed = overflow_mask[:, 0].ravel()  # (N, ) same mask but only for seed particles

    # First part is True when corresponding xnk or vnk is inf due to overflow
    # Second part is True when squaring vnk leads to overflow
    max_invmass = inv_mass_diag_next.max()  # maximum of the inverse mass diagonal
    ofm = overflow_mask.ravel() | np.any(np.abs(vnk.reshape(-1, d)) >= np.sqrt(np.finfo(np.float64).max), axis=1)  # (N*(T+1), )
    # same mask but only for seed particles
    ofm_seed = overflow_mask[:, 0].ravel() | np.any(np.abs(vnk[:, 0]) >= np.sqrt(np.finfo(np.float64).max), axis=1)  # (N, )

    log_num = np.repeat(-np.inf, N*Tplus1)  # default to zero denominator
    log_den = np.repeat(np.nan, N)  # no overflown particle can ever become a seed

    # Log numerator of the unfolded weights
    log_num[~ofm] = (-nlps.ravel()[~ofm]) + gamma_next*(-nlls.ravel()[~ofm])
    log_num[~ofm] -= 0.5*np.sum(inv_mass_diag_next * vnk.reshape(-1, d)[~ofm]**2, axis=1)
    log_num[~ofm] += 0.5*np.sum(np.log(inv_mass_diag_next))

    # Log Denominator of the unfolded weights
    log_den[~ofm_seed] = (-nlps[~ofm_seed, 0]) + gamma_curr*(-nlls[~ofm_seed, 0])
    log_den[~ofm_seed] -= 0.5*np.sum((vnk[~ofm_seed, 0]**2) * inv_mass_diag_curr, axis=1)
    log_den[~ofm_seed] += 0.5*np.sum(np.log(inv_mass_diag_curr))

    # Unfolded weights
    logw_unfolded = log_num.reshape(N, Tplus1) - log_den[:, None]  # (N, T+1) log un-normalized unfolded weights
    W_unfolded = np.exp(logw_unfolded - logsumexp(logw_unfolded))  # (N, T+1) normalized unfolded weights

    # Overflown should lead to zero weights
    if W_unfolded[overflow_mask].sum() != 0:
        raise ValueError(f"Weights should be zero but they are {W_unfolded[overflow_mask].sum()}")

    # Folded weights
    logw_folded = logsumexp(logw_unfolded, axis=1) - np.log(Tplus1)  # (N, ) un-normalized folded weights
    W_folded = np.exp(logw_folded - logsumexp(logw_folded))  # (N, ) normalized folded weights

    return W_unfolded, logw_unfolded, W_folded, logw_folded


def estimate_with_cond_variance(xnk: NDArray, logw: NDArray, epsilons: NDArray, ss: callable, skip_overflown: bool,
                                overflow_mask: NDArray) -> Tuple:
    """Estimates a sufficient statistics using the conditional variance.

    Parameters
    ----------
    :param xnk: Snippet positions, has shape `(N, T+1, d)` where `d` is the dimension of the position space
    :type xnk: np.ndarray
    :param logw: Log un-normalized unfolded weights, has shape `(N, T+1)`
    :type logw: np.ndarray
    :param epsilons: Step sizes, one for each snippet, has shape `(N, )`
    :type epsilons: np.ndarray
    :param skip_overflown: Whether to skip the trajectories that had an epsilon that lead to overflow. Shape (N, )
    :type skip_overflown: bool
    :param overflow_mask: Mask determining if the snippet has encoutered an overflow error
    :type overflow_mask: np.ndarray
    :return: Dictionary of estimates for each sufficient statistics
    :rtype: dict
    """
    assert len(xnk.shape) == 3, "Snippet positions must have three dimensions `(N, T+1, d)`."
    assert len(logw.shape) == 2, "Log un-normalized unfolded weights must have two dimensions `(N, T+1)`."
    assert len(epsilons.shape) == 1, "Epsilons must be one dimensional `(N, )`."
    assert xnk.shape[0] == len(epsilons), "Number of snippets must match the number of epsilons."
    assert xnk.shape[:2] == logw.shape, "There must be a weight for each snippet position."

    # Weight computations are in common for all sufficient statistics
    N, Tplus1 = logw.shape
    # ofm = overflow_mask if skip_overflown else np.zeros(N, dtype=bool)
    # Compute the discrete distribution mu(k | z, epsilon)
    # mu_k_given_z_eps = np.exp(logw[~ofm] - logsumexp(logw[~ofm], axis=1, keepdims=True))  # (N, T+1)
    mu_k_given_z_eps = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))  # (N, T+1)
    # Compute the conditional expectation for the position function mu(f| z, epsilon)
    cond_exp = np.sum(xnk * mu_k_given_z_eps[:, :, None], axis=1, keepdims=True)  # (N, 1, d)
    # Compute the squared norms
    norms = np.linalg.norm(xnk - cond_exp, axis=2)  # (N, T+1)
    # Compute the test statistics for each snippet/epsilon
    T_hat = ss(epsilons)[:, None]   # (N, 1)
    # Compute terms of the form np.sum(sq_norms * w * T_eps) in a way to avoid numerical errors on the log scale
    T_hat_repeated = np.tile(T_hat, Tplus1)  # (N, T+1)
    # Flag for when computation can be "logged"
    flag = (norms > 0) & (logw > -np.inf) & (T_hat_repeated != 0)  # (N, T+1) when computation is not zero
    # Initialize variables
    log_sq_norms = np.full(fill_value=-np.inf, shape=(N, Tplus1))
    logw_filtered = np.full(fill_value=-np.inf, shape=(N, Tplus1))
    log_T_hat = np.full(fill_value=-np.inf, shape=(N, Tplus1))
    # log of squared norms, computed only where computation is not zero
    log_sq_norms[flag] = 2*np.log(norms[flag])
    # Same for log weights and estimator
    logw_filtered[flag] = logw[flag]  # (n_filtered, )
    log_T_hat[flag] = np.log(T_hat_repeated[flag])    # (n_filtered, )
    log_criterion = log_sq_norms + logw_filtered  # (N, T+1)
    # Compute scalar estimator
    estimator = np.exp(
        logsumexp(log_criterion[flag] + log_T_hat[flag]) - logsumexp(log_criterion[flag])
    )
    return estimator, log_criterion, flag


def hamiltonian_snippet(N: int, T: int, mass_diag: NDArray, ESSrmin: float, sample_prior: callable,
                        compute_likelihoods_priors_gradients: callable, epsilon_params: dict,
                        act_on_overflow: bool = False, adapt_step_size: bool = False,
                        skip_overflown: bool = False, adapt_mass: bool = False, verbose: bool = True,
                        seed: Optional[int] = None):
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
    :param skip_overflown: Whether to skip overflown trajectories when estimating epsilon
    :type skip_overflown: bool
    :param adapt_mass: Whether to adapt the mass matrix diagonal or not
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
    mass_diag_curr = mass_diag if mass_diag is not None else np.eye(d)
    v = rng.normal(loc=0, scale=1, size=(N, d)) * np.sqrt(mass_diag_curr)

    # Initial step sizes and mass matrix
    epsilons = sample_epsilons(eps_params=epsilon_params, N=N, rng=rng)

    # Storage
    epsilon_history = [epsilons]
    epsilon_params_history = [{key: value for key, value in epsilon_params.items() if type(value) != callable}]  # parameters for the epsilon distribution
    gammas = [0.0]
    ess_history = [N]
    logLt = 0.0

    while gammas[n-1] < 1.0:
        verboseprint(f"Iteration {n} Gamma {gammas[n-1]: .5f} Epsilon {epsilon_params['to_print'].capitalize()}: {epsilon_params[epsilon_params['to_print']]}")

        # Construct trajectories
        xnk, vnk, nlps, nlls = leapfrog(x, v, T, epsilons, gammas[n-1], 1/mass_diag_curr, compute_likelihoods_priors_gradients)
        verboseprint("\tTrajectories constructed.")

        # Check if there is any inf due to overflow error
        if not (np.all(np.isfinite(xnk)) and np.all(np.isfinite(vnk)) and
                np.all(np.isfinite(nlps)) and np.all(np.isfinite(nlls))):
            overflow_mask = np.any(~np.isfinite(xnk), axis=2) | np.any(~np.isfinite(vnk), axis=2) | ~np.isfinite(nlps) | ~np.isfinite(nlls) # (N, T+1)
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
            W_mass_est, _, _, _ = compute_weights(
                vnk, nlps, nlls, 1/mass_diag_curr, 1/mass_diag_curr, gammas[n], gammas[n-1],
                overflow_mask=overflow_mask
            )
            weighted_mean = np.average(xnk.reshape(-1, d)[~overflow_mask.ravel()], axis=0, weights=W_mass_est.ravel()[~overflow_mask.ravel()])  # (d, )
            mass_diag_next = 1 / np.average((xnk.reshape(-1, d)[~overflow_mask.ravel()] - weighted_mean)**2, axis=0, weights=W_mass_est.ravel()[~overflow_mask.ravel()])
            verboseprint(f"\tNew mass matrix diagonal estimated. Mean {mass_diag_next.mean()}")
        else:
            mass_diag_next = mass_diag_curr

        # Compute weights and ESS
        W_unfolded, logw_unfolded, W_folded, logw_folded = compute_weights(
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
            old_mean = epsilon_params['mean']
            old_lambda = 9*old_mean / epsilon_params['skewness']**2
            estimate, lala, flag = estimate_with_cond_variance(xnk=xnk, logw=logw_unfolded, epsilons=epsilons,
                                                               ss=lambda e: e, skip_overflown=False,
                                                               overflow_mask=overflow_mask)
            epsilon_params['mean'] = estimate
            new_lambda = 9*estimate / epsilon_params['skewness']**2
            try:
                upsilons = logsumexp(lala.reshape(-1, T+1), axis=1)
                # we just do this to have a better figure (maybe)
                epsilons_fake = np.linspace(epsilons.min(), 5*epsilons.max(), N)
                xnk_fake, vnk_fake, nlps_fake, nlls_fake = leapfrog(xnk[:, 0], vnk[:, 0], T, epsilons_fake, gammas[n-1], 1/mass_diag_curr, compute_likelihoods_priors_gradients)
                overflow_mask_fake = np.any(~np.isfinite(xnk_fake), axis=2) | np.any(~np.isfinite(vnk_fake), axis=2) | ~np.isfinite(nlps_fake) | ~np.isfinite(nlls_fake)
                _, logw_unfolded_fake, _, _ = compute_weights(vnk_fake, nlps_fake, nlls_fake, 1/mass_diag_next, 1/mass_diag_curr, gammas[n], gammas[n-1], overflow_mask=overflow_mask_fake)
                _, lala_fake, _ = estimate_with_cond_variance(xnk=xnk_fake, logw=logw_unfolded_fake,
                                                                       epsilons=epsilons_fake, ss=lambda e: e,
                                                                       skip_overflown=False, overflow_mask=overflow_mask_fake)
                upsilons_fake = logsumexp(lala_fake.reshape(-1, T+1), axis=1)
                shifted_upsilons_fake = upsilons_fake - upsilons_fake.min()
                # import matplotlib.pyplot as plt
                # sample the new ones
                new_epsilons = sample_epsilons(epsilon_params, N=N, rng=rng)
                # import seaborn as sns
                shifted_upsilons = upsilons #- upsilons.min()
                fig, ax = plt.subplots(figsize=(4, 4), sharex=True, sharey=True)
                # ax.scatter(epsilons, upsilon, alpha=0.5)
                counts, bins, bars = ax.hist(epsilons, bins=50, density=True, color='lightcoral', ec='brown', zorder=10, label=r'$\mathregular{\nu_{n-1}}$')
                _ = [b.remove() for b in bars]
                # sns.kdeplot(epsilons, color='lightcoral', ax=ax[0], label=r'$\mathregular{\nu_{n-1}}$')
                # sns.kdeplot(new_epsilons, color='lightseagreen', ax=ax[0], label=r'$\mathregular{\nu_{n}}$')
                xaxis_vals = np.linspace(min(epsilons.min(), new_epsilons.min()), 1.1*max(epsilons.max(), new_epsilons.max()), 1000)
                # ax[0].plot(xaxis_vals, invgauss(mu=old_mean/old_lambda, loc=0, scale=old_lambda).pdf(xaxis_vals), color='lightcoral')
                # ax[0].plot(xaxis_vals, invgauss(mu=estimate/new_lambda, loc=0, scale=new_lambda).pdf(xaxis_vals), color='lightseagreen')
                #ax[1].hist2d(epsilons, (shifted_upsilons / shifted_upsilons.max()) * (counts.max()), bins=30, zorder=0, label='upsilon')
                # ax[1].hist2d(epsilons_fake, (shidted_upsilons_fake / shidted_upsilons_fake.max()) * counts.max(), bins=30, cmap='Blues', zorder=0)
                hist2d = ax.hist2d(epsilons, (shifted_upsilons / shifted_upsilons.max()) * (1.5*counts.max()), bins=30, zorder=0, cmap='Blues')
                # ax[1].scatter(epsilons_fake, (shidted_upsilons_fake / shidted_upsilons_fake.max()) * counts.max(), alpha=0.5, zorder=0, color='gold', ec='goldenrod')
                # ax[1].scatter(epsilons, (shifted_upsilons / shifted_upsilons.max()) * (counts.max()), alpha=0.5, zorder=1, color='thistle', ec='violet')
                ax2 = ax.twinx()
                old_dens = ax2.plot(xaxis_vals, invgauss(mu=old_mean/old_lambda, loc=0, scale=old_lambda).pdf(xaxis_vals), color='gold', zorder=1, lw=2, label=r'$\mathregular{\nu_{n-1}}$')
                new_dens = ax2.plot(xaxis_vals, invgauss(mu=estimate/new_lambda, loc=0, scale=new_lambda).pdf(xaxis_vals), color='indianred', zorder=2, lw=2, label=r'$\mathregular{\nu_{n}}$')
                ax.set_xlabel(r"$\mathregular{\epsilon}$", fontsize=13)
                ax.set_ylabel(r'$\mathregular{\upsilon_n(\epsilon, z)}$', fontsize=13) #"Criterion")
                ax2.set_ylabel("Density", fontsize=13)
                proxy_hist2d = mpatches.Patch(color='cornflowerblue', label=r'$\mathregular{\upsilon_n(\epsilon, z)}$')
                handles, labels = ax2.get_legend_handles_labels()
                # handles.append(proxy_hist2d)
                # ax[1].scatter(epsilons, (shifted_upsilons / shifted_upsilons.max()) * (hist_out[0].max()), alpha=0.5, zorder=0, label='upsilon')
                # new_hist_out = ax[2].hist(new_epsilons, bins=50, density=True, color='lightseagreen', ec='teal', zorder=8,  label=r'$\mathregular{\nu_{n}}$')
                # sns.kdeplot(epsilons, ax=ax)            #ax.hist2d(epsilons, absolute_upsilons, bins=30, cmap='Blues', zorder=0, label=r'$\mathregular{\upsilon(\epsilon, z))}$')
                #ax.scatter(epsilons, absolute_upsilons, alpha=0.5)
                # ax.set_xscale('log')
                plt.tight_layout()
                ax2.legend(handles=handles)
                plt.colorbar(hist2d[3], ax=ax, label='Counts', orientation='horizontal', location='top')
                plt.show()
            except ValueError:
                print("\tSkipping plot.")
        # step_size = estimate_new_epsilon_mean(xnk=xnk, logw=logw_unfolded, epsilons=epsilons, ss=lambda _eps: _eps)
        epsilons = sample_epsilons(eps_params=epsilon_params, N=N, rng=rng)
        verboseprint(f"\tEpsilon {epsilon_params['to_print'].capitalize()} {epsilon_params[epsilon_params['to_print']]}")

        # Storage
        epsilon_history.append(epsilons)
        epsilon_params_history.append({key: value for key, value in epsilon_params.items() if type(value) != callable})
        ess_history.append(ess)

        n += 1
    runtime = time.time() - start_time
    return {"logLt": logLt, "gammas": gammas, "runtime": runtime, "epsilons": epsilon_history, "ess": ess_history,
            'epsilon_params_history': epsilon_params_history}


if __name__ == "__main__":
    # Grab data
    data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonar.npy"))
    y = -data[:, 0]  # Shape (208,)
    Z = data[:, 1:]  # Shape (208, 61)
    scales = np.array([5] * 61)
    scales[0] = 20

    # Define function to sample the prior
    sample_prior = generate_sample_prior_function(_scales=scales)
    compute_likelihoods_priors_gradients = generate_nlp_gnlp_nll_and_gnll_function(_y=y, _Z=Z, _scales=scales)

    # Run Hamiltonian snippets
    n_runs = 20
    overall_seed = np.random.randint(low=0, high=10000000000)
    seeds = np.random.default_rng(overall_seed).integers(low=1, high=10000000000, size=n_runs)
    step_sizes = [0.001] #np.array(np.geomspace(start=0.001, stop=10.0, num=9))  # np.array() used only for pylint
    N = 1000
    T = 30
    skewness = 1  # a large skewness helps avoiding a large bias
    mass_matrix_adaptation = False
    mass_diag = 1 / scales**2 if mass_matrix_adaptation else np.ones(61)
    verbose = False
    skipo = False
    aoo = False
    adapt_step_size = True

    results = []
    for i in range(n_runs):
        print(f"Run: {i}")
        for eps_ix, eps in enumerate(step_sizes):
            epsilon_params = {
                'distribution': 'inv_gauss',
                'skewness': skewness,
                'mean': eps,
                'params_to_estimate': {'mean': lambda epsilon: epsilon},
                'to_print': 'mean',
                'on_overflow': lambda param_dict: {'skewness': max(1, param_dict['skewness'] * 0.99)}
            }
            res = {'N': N, 'T': T, 'epsilon_params': epsilon_params}
            out = hamiltonian_snippet(N=N, T=T, mass_diag=mass_diag, ESSrmin=0.8,
                                      sample_prior=sample_prior,
                                      epsilon_params=epsilon_params,
                                      act_on_overflow=aoo,
                                      compute_likelihoods_priors_gradients=compute_likelihoods_priors_gradients,
                                      adapt_mass=mass_matrix_adaptation, skip_overflown=skipo,
                                      adapt_step_size=adapt_step_size,
                                      verbose=verbose, seed=seeds[i])
            res.update({'logLt': out['logLt'], 'out': out})
            print(f"\t\tEps: {eps: .7f} \tLogLt: {out['logLt']: .1f} \tFinal ESS: {out['ess'][-1]: .1f}"
                  f"\tEps {epsilon_params['to_print'].capitalize()}: "
                  f"{out['epsilon_params_history'][-1][epsilon_params['to_print']]: .3f} Seed {int(seeds[i])} ")
            results.append(res)
    #
    # # Save results
    # # with open(f"logistic_regression/results/seed{overall_seed}_N{N}_T{T}_mass{mass_matrix_adaptation}_runs{n_runs}_from{eps_to_str(min(step_sizes))}_to{eps_to_str(max(step_sizes))}_skewness{skewness}.pkl", "wb") as file:
    # #     pickle.dump(results, file)
