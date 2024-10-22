import numpy as np
import pickle
import os
import time
from typing import Optional
from scipy.special import logsumexp
from numpy.typing import NDArray
from scipy.stats import invgauss
import matplotlib.pyplot as plt
from matplotlib import rc
from typing import Tuple
from scipy.optimize import brentq
from copy import deepcopy
rc('font', **{'family': 'STIXGeneral'})


def adapt_num_leapfrog_steps_contractivity(
        xnk: NDArray, vnk: NDArray, epsilons: NDArray, nlps: NDArray, nlls: NDArray, T: int,
        gamma: float, mass_params: dict, compute_likelihoods_priors_gradients: callable,
        rng: np.random.Generator, max_tries: int = 100, plot_contractivity: bool = False, T_max: int = 100,
        T_min: int = 5) -> Tuple[int, bool]:
    """Adapts the number of leapfrog steps using a contractivity argument.

    Parameters
    ----------
    :param xnk: Positions of the trajectories, jointly with vnk they are distributed according to \\bar{mu}_{n-1}
    :type xnk: np.array
    :param vnk: Velocities of the trajectories, jointly with xnk they are distributed according to \\bar{mu}_{n-1}
    :type vnk: np.array
    :param epsilons: Step sizes sampled from \\nu_{n-1}
    :type epsilons: np.array
    :param nlps: Negative log priors for the positions xnk
    :type nlps: np.array
    :param nlls: Negative log likelihoods for the positions xnk
    :type nlls: np.array
    :param T: Number of leapfrog steps used to construct xnk and vnk
    :type T: int
    :param gamma: Current tempering parameter \\gamma_{n-1}
    :type gamma: float
    :param mass_params: Mass matrix parameters
    :type mass_params: dict
    :param compute_likelihoods_priors_gradients: Function computing nlps, gnlps, nlls, gnlls
    :type compute_likelihoods_priors_gradients: callable
    :param rng: Random number generator for reproducibility
    :type rng: np.random.generator
    :param max_tries: Number of maximum attempts at finding a coupling
    :type max_tries: int
    :param plot_contractivity: Whether to plot the contractivity
    :type plot_contractivity: bool
    :param T_max: Maximum number of integration steps allowed
    :type T_max: int
    :param T_min: Minimum number of integration steps allowed
    :type T_min: int
    :return: A tuple (T, coupling_found) where T is the optimal num of leapfrog steps and the other a boolean flag
    :rtype: tuple
    """
    assert isinstance(max_tries, int) and max_tries >= 1, "Maximum number of tries should be an integer >= 1."
    assert isinstance(T_min, int) and T_min >= 1, "Minimum number of integration steps must be an integer >= 1."
    N, Tplus1, d = xnk.shape

    # Couple velocities and epsilons
    v_coupled, eps_coupled = vnk[:, 0], epsilons  # initialize as they are
    coupling_found = False
    try_number = 0
    coupling = None
    while try_number <= max_tries:  # find a coupling where coupled positions are different
        coupling = rng.permutation(x=N).reshape(-1, 2)
        if np.any(np.linalg.norm(xnk[coupling[:, 1], 0] - xnk[coupling[:, 0], 0], axis=-1) == 0):
            try_number += 1
        else:
            coupling_found = True
            break
    if coupling_found:

        v_coupled[coupling[:, 1]] = vnk[coupling[:, 0], 0]  # coupled velocities are identical
        eps_coupled[coupling[:, 1]] = epsilons[coupling[:, 0]]  # coupled epsilons are identical

        # Construct trajectories using duplicated velocities and step sizes
        xnk[coupling[:, 1]], vnk[coupling[:, 1]], nlps[coupling[:, 1]], nlls[coupling[:, 1]] = leapfrog(
            x=xnk[coupling[:, 1], 0],
            v=v_coupled[coupling[:, 1]],
            T=T,
            epsilons=eps_coupled[coupling[:, 1]],
            gamma_curr=gamma,
            mass_params=mass_params,
            compute_likelihoods_priors_gradients=compute_likelihoods_priors_gradients
        )

        # Check for overflow errors on xnk just generated
        ofm0 = np.any(~np.isfinite(xnk[coupling[:, 0]]), axis=2)  # (N//2, T+1) True when particle overflown
        ofm1 = np.any(~np.isfinite(xnk[coupling[:, 1]]), axis=2)  # (N//2, T+1) True when particle overflown
        ofm = ofm0 | ofm1  # (N//2, T+1) True when either of the coupled particles overflown

        # Compute contractivity between coupled trajectories
        contractivity = np.full(fill_value=np.inf, shape=(N//2, Tplus1))
        contractivity[~ofm] = np.linalg.norm(xnk[coupling[:, 0]][~ofm] - xnk[coupling[:, 1]][~ofm], axis=1)  # (N//2, T+1)
        contractivity /= contractivity[:, 0].reshape(-1, 1)
        contractivity = np.clip(contractivity, a_min=None, a_max=5)  # to avoid coupled particles diverging too much

        # Find left tail of contractivity distribution
        bottom_quantile_val = 0.1
        bottom_contractivity_flag = contractivity <= np.quantile(contractivity, q=bottom_quantile_val)

        # Find the i-indices and k-indices corresponding to tail contraction
        bottom_i, bottom_k = np.where(bottom_contractivity_flag)
        bottom_taus = eps_coupled[coupling[:, 0]][bottom_i] * bottom_k  # corresponding integration times

        # Integration times corresponding to coupled particles
        taus = np.arange(Tplus1).reshape(1, -1) * eps_coupled[coupling[:, 0]].reshape(-1, 1)  # (N//2, T+1)

        # Compute a binned average of the contractivity as a function of tau
        n_bins = max(100, np.sqrt(N))
        avg_contraction, tau_bins = binned_average(contractivity, taus, n_bins)  # (n_bins, ), (n_bins, )
        tau_min_contraction = tau_bins[np.nanargmin(avg_contraction)]  # tau corresponding to the minimum average contraction
        median_eps_bottom = np.quantile(eps_coupled[coupling[:, 0]][bottom_i], q=0.5)  # median epsilon corresponding to tail of contraction distribution
        bottom_taus_median = np.quantile(bottom_taus, q=0.5)  # median of taus corresponding to tail of contraction distribution

        # Compute the new T as the tau corresponding to the average min contraction divided by the median step size among particles in the tail of the contraction distribution
        T_optimal = np.ceil(tau_min_contraction/median_eps_bottom).astype(int)

        if plot_contractivity:
            fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
            # Subplot 1: contractivity as a function of tau
            ax[0].plot(taus.T, contractivity.T, color='lightcoral', alpha=0.5)
            ax[0].plot(tau_bins, avg_contraction, color='black')
            ax[0].axvline(tau_min_contraction, color='navy', ls='--', label=f'Min-Contractivity Tau {tau_min_contraction: .3f}')
            ax[0].set_xlabel("Tau")
            ax[0].set_ylabel("Contractivity")
            ax[0].legend()
            # Subplot 2: 2D histogram of epsilon and k corresponding to tail of contractivity distribution
            ax[1].hist2d(eps_coupled[coupling[:, 0]][bottom_i], bottom_k, bins=30)
            ax[1].set_xlabel(r"$\mathregular{\epsilon}$")
            ax[1].set_ylabel(r"$\mathregular{k}$")
            ax[1].set_title(f"Median eps {median_eps_bottom: .4f}, Resulting T {T_optimal}")
            # Subplot 2: Histogram of bottom taus
            _ = ax[2].hist(bottom_taus, bins=Tplus1)
            ax[2].axvline(bottom_taus_median, label=f'Median {bottom_taus_median: .3f}', ls='--', lw=2, color='black', zorder=1)
            ax[2].set_xlabel("Taus")
            ax[2].set_ylabel("Counts")
            ax[2].legend()
            ax[2].set_title(f"Taus for bottom {bottom_quantile_val} contractivity quantile", fontsize=10)
            fig.suptitle(f"T {T} epsilon mean {epsilons.mean(): .4f}")
            plt.tight_layout()
            plt.show()
        T = max(min(T_max, T_optimal), T_min)
    return T, coupling_found


def binned_average(y_values: NDArray, x_values: NDArray, n_bins: int) -> Tuple[NDArray, NDArray]:
    """Computes a binned average of the squared norms over the taus.

    Parameters
    ----------
    :param y_values: Squared norms of the coupled particles, should have shape (n, T+1) where n <= N
    :type y_values: np.ndarray
    :param x_values: Integration times for the coupled particles, should have shape (n, T+1) where n <= N
    :type x_values: np.ndarray
    :param n_bins: Number of bins to divide the integration time into
    :type n_bins: int
    :return: tuple of binned mean of squared norms and tau bins used
    :rtype: tuple
    """
    # Divide the integration time into bins
    x_bins = np.linspace(start=x_values.min(), stop=x_values.max(), num=n_bins)
    bin_indices = np.digitize(x_values, x_bins, right=True)  # since taus will include zeros
    binned_means = np.full(fill_value=np.nan, shape=n_bins)
    for i in range(n_bins):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            binned_means[i] = np.mean(y_values[mask])
    return binned_means, x_bins


def estimate_with_cond_variance(xnk: NDArray, logw_criterion: NDArray, epsilons: NDArray, epsilon_params: dict, skip_overflown: bool,
                                overflow_mask: NDArray, plot_Q_criterion: bool) -> NDArray:
    """Estimates a sufficient statistics using the conditional variance.

    Parameters
    ----------
    :param xnk: Snippet positions, has shape `(N, T+1, d)` where `d` is the dimension of the position space
    :type xnk: np.ndarray
    :param logw_criterion: Log un-normalized criterion weights, has shape `(N, T+1)`, where are not the unfolded weights
    :type logw_criterion: np.ndarray
    :param epsilons: Step sizes, one for each snippet, has shape `(N, )`
    :type epsilons: np.ndarray
    :param epsilon_params: Dictionary of epsilon parameters
    :type epsilon_params: dict
    :param skip_overflown: Whether to skip the trajectories that had an epsilon that lead to overflow. Shape (N, )
    :type skip_overflown: bool
    :param overflow_mask: Mask determining if the snippet has encountered an overflow error
    :type overflow_mask: np.ndarray
    :param plot_Q_criterion: Whether to plot the Q criterion
    :type plot_Q_criterion: bool
    :return: Dictionary of estimates for each sufficient statistics
    :rtype: dict
    """
    assert len(xnk.shape) == 3, "Snippet positions must have three dimensions `(N, T+1, d)`."
    assert len(logw_criterion.shape) == 2, "Log un-normalized unfolded weights must have two dimensions `(N, T+1)`."
    assert len(epsilons.shape) == 1, "Epsilons must be one dimensional `(N, )`."
    assert xnk.shape[0] == len(epsilons), "Number of snippets must match the number of epsilons."
    assert xnk.shape[:2] == logw_criterion.shape, "There must be a weight for each snippet position."

    # Weight computations are in common for all sufficient statistics
    N, Tplus1 = logw_criterion.shape
    ofm = overflow_mask if skip_overflown else np.zeros(N, dtype=bool)

    # Compute the discrete distribution mu(k | z, epsilon)
    mu_k_given_z_eps = np.exp(logw_criterion[~ofm] - logsumexp(logw_criterion[~ofm], axis=1, keepdims=True))  # (N, T+1)
    # Compute the conditional expectation for the position function mu(f| z, epsilon)
    cond_exp = np.sum(xnk[~ofm] * mu_k_given_z_eps[:, :, None], axis=1, keepdims=True)  # (N, 1, d)
    # Compute the squared norms
    norms = np.linalg.norm(xnk[~ofm] - cond_exp, axis=2)  # (N, T+1)
    # Base flag
    base_flag = (norms > 0) & (logw_criterion[~ofm] > -np.inf)  # (N, T+1)

    # Fill up Qs for all non-overflown particles, independently of which epsilon they used
    Qs = np.full(fill_value=0.0, shape=(N, Tplus1))
    Qs[base_flag] = np.exp(logw_criterion[base_flag]) * (norms[base_flag]**2)
    # Now for each epsilon, compute the average
    Q_vals = np.full(fill_value=np.nan, shape=len(epsilon_params['values']))
    for _eps_ix, _eps in enumerate(epsilon_params['values']):
        Q_vals[_eps_ix] = Qs[epsilons == _eps].sum(axis=1).mean()
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.plot(epsilon_params['values'], Q_vals, marker='o', color='lightcoral', markeredgecolor='firebrick', lw=2)
    # ax.set_xscale('log')
    # ax.set_xlabel(r"$\mathregular{\epsilon}$", fontsize=14)
    # ax.set_ylabel(r"$\mathregular{\widehat{\mathcal{Q}}_{\gamma_{n-1}, T}}$", fontsize=16)
    # ax.grid(True, color='gainsboro')
    # ax.set_yscale('log')
    # # ax.legend()
    # plt.show()

    # estimators = {key: 0 for key in epsilon_params['params_to_estimate'].keys()}
    # if epsilon_params['distribution'] != "discrete_uniform":
    #     for param_name, ss in epsilon_params['params_to_estimate'].items():
    #         # Compute the test statistics for each snippet/epsilon
    #         T_hat = ss(epsilons[~ofm])[:, None]   # (N, 1)
    #         # Compute terms of the form np.sum(sq_norms * w * T_eps) in a way to avoid numerical errors on the log scale
    #         T_hat_repeated = np.tile(T_hat, Tplus1)  # (N, T+1)
    #         # Flag for when computation can be "logged"
    #         flag = base_flag & (T_hat_repeated != 0)  # (N, T+1) when computation is not zero
    #         # log of squared norms, computed only where computation is not zero
    #         log_sq_norms = 2*np.log(norms[flag])
    #         # Same for log weights and estimator
    #         logw_filtered = logw_criterion[~ofm][flag]  # (n_filtered, )
    #         log_T_hat = np.log(T_hat_repeated[flag])    # (n_filtered, )
    #         # Compute scalar estimator
    #         estimator = np.exp(
    #             logsumexp(log_sq_norms + logw_filtered + log_T_hat) - logsumexp(log_sq_norms + logw_filtered)
    #         )
    #         estimators[param_name] = estimator
    # else:
    #     estimators = {'values': epsilon_params['values']}
    estimators = {'values': epsilon_params['values']}
    return estimators, Q_vals

def compute_weights(
        vnk: NDArray,
        nlps: NDArray,
        nlls: NDArray,
        mass_params: dict,
        gamma_next: float,
        gamma_curr: float,
        overflow_mask: NDArray,
        computation_for_mass_matrix_adaptation: bool = False
        ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Computes unfolded and folded weights.

    Parameters
    ----------
    :param vnk: Velocities constructed with $\\psi_{n-1}$, has shape `(N, T+1, d)` where `d` is the velocity dimension
    :type vnk: numpy.ndarray
    :param nlps: Negative log prior density on the snippets, has shape `(N, T+1)`
    :type nlps: numpy.ndarray
    :param nlls: Negative log likelihood on the snippets, has shape `(N, T+1)`
    :type nlls: numpy.ndarray
    :param mass_params: Mass matrix parameters
    :type mass_params: dict
    :param gamma_next: Tempering parameter $\\gamma_n$ for the next distribution, will be used in the numerator
    :type gamma_next: float
    :param gamma_curr: Tempering parameter $\\gamma_{n-1}$ for the current distribution, will be used in the denominator
    :type gamma_curr: float
    :param overflow_mask: Mask indicating where there has been an overflow either in xnk or vnk. There we do not
                          compute a weight, it will be set to zero. Will have shape (N, T+1).
    :type overflow_mask: numpy.ndarray
    :param computation_for_mass_matrix_adaptation: If True, we use varpi_{n-1} in the numerator too
    :type computation_for_mass_matrix_adaptation: bool
    :return: Tuple containing `(W_unfolded, logw_unfolded, W_folded, logw_unfolded, logw_criterion)`
    :rtype: tuple(numpy.array, numpy.array, numpy.array, numpy.array, numpy.array)
    """
    assert len(vnk.shape) == 3, "Velocities must be a 3-dimensional array."
    assert len(nlps.shape) == 2, "Negative log priors must be a 2-dimensional array."
    assert len(nlls.shape) == 2, "Negative log likelihoods must be a 2-dimensional array."
    N, Tplus1, d = vnk.shape

    # First part is True when corresponding xnk or vnk is inf due to overflow
    # Second part is True when squaring vnk leads to overflow
    # max_invmass = inv_mass_diag_next.max()  # maximum of the inverse mass diagonal
    ofm = overflow_mask.ravel() | np.any(np.abs(vnk.reshape(-1, d)) >= np.sqrt(np.finfo(np.float64).max), axis=1)  # (N*(T+1), )
    # same mask but only for seed particles
    ofm_seed = overflow_mask[:, 0].ravel() | np.any(np.abs(vnk[:, 0]) >= np.sqrt(np.finfo(np.float64).max), axis=1)  # (N, )

    log_num = np.repeat(-np.inf, N*Tplus1)  # default to zero denominator
    log_num_unfolded = np.repeat(-np.inf, N*Tplus1)
    log_num_criterion = np.repeat(-np.inf, N*Tplus1)
    log_den = np.repeat(np.nan, N)  # no overflown particle can ever become a seed

    # Log numerator of the unfolded weights
    log_num[~ofm] = (-nlps.ravel()[~ofm])
    log_num[~ofm] += compute_quadratic_form(mass_params, vnk.reshape(-1, d)[~ofm], numerator_computation=not computation_for_mass_matrix_adaptation)
    log_num[~ofm] += compute_determinant_term(mass_params, numerator_computation=not computation_for_mass_matrix_adaptation)

    log_num_unfolded[~ofm] = log_num[~ofm] + gamma_next*(-nlls.ravel()[~ofm])
    log_num_criterion[~ofm] = log_num[~ofm] + gamma_curr*(-nlls.ravel()[~ofm])

    # Log Denominator of the unfolded weights
    log_den[~ofm_seed] = (-nlps[~ofm_seed, 0]) + gamma_curr*(-nlls[~ofm_seed, 0])
    log_den[~ofm_seed] += compute_quadratic_form(mass_params, vnk[~ofm_seed, 0], numerator_computation=False)
    log_den[~ofm_seed] += compute_determinant_term(mass_params, numerator_computation=False)

    # Unfolded weights
    logw_unfolded = log_num_unfolded.reshape(N, Tplus1) - log_den[:, None]  # (N, T+1) log un-normalized unfolded weights
    W_unfolded = np.exp(logw_unfolded - logsumexp(logw_unfolded))  # (N, T+1) normalized unfolded weights

    # Weights for criterion
    logw_criterion = log_num_criterion.reshape(N, Tplus1) - log_den[:, None]  # (N, T+1) similar to unfolded

    # Overflown should lead to zero weights
    if W_unfolded[overflow_mask].sum() != 0:
        raise ValueError(f"Weights should be zero but they are {W_unfolded[overflow_mask].sum()}")

    # Folded weights
    logw_folded = logsumexp(logw_unfolded, axis=1) - np.log(Tplus1)  # (N, ) un-normalized folded weights
    W_folded = np.exp(logw_folded - logsumexp(logw_folded))  # (N, ) normalized folded weights

    return W_unfolded, logw_unfolded, W_folded, logw_folded, logw_criterion


def compute_quadratic_form(mass_params: dict, v: NDArray, numerator_computation: bool = True) -> NDArray:
    """Computes the quadratic form for the unfolded weights.

    Parameters
    ----------
    :param mass_params: Parameters for the mass matrix
    :type mass_params: dict
    :param v: Velocities to compute the quadratic form for
    :type v: np.ndarray
    :param numerator_computation: Whether the computation is for the numerator or denominator of the unfolded weights
    :type numerator_computation: bool
    :return: Quadratic form for a gaussian N(0, M) for each velocity provided
    :rtype: np.ndarray
    """
    suffix = "next" if numerator_computation else "curr"
    match mass_params["strategy"], mass_params["matrix_type"]:
        case ("fixed" | "schedule" | "adaptive"), "diag":
            return -0.5*np.sum(mass_params[f"mass_diag_{suffix}"] * v**2, axis=1)
        case ("fixed" | "schedule"), "full":
            M_inv_v = np.linalg.solve(mass_params[f"mass_{suffix}"], v.T).T
            return -0.5*np.einsum('ij,ij->i', v, M_inv_v)
        case "adaptive", "full":
            raise NotImplemented("Mass Matrix adaptation with a full mass matrix is not implemented yet.")


def compute_determinant_term(mass_params: dict, numerator_computation: bool) -> float:
    """Computes the determinant term for the numerator or denominator of the unfolded weights.

    Parameters
    ----------
    :param mass_params: Parameters of the mass matrix
    :type mass_params: dict
    :param numerator_computation: Whether the computation is for the numerator or denominator of the unfolded weights
    :type numerator_computation: bool
    :return: Log determinant term for a gaussian with covariance matrix equal to the mass matrix in the mass_params
    :rtype: float

    Notes
    -----
    It computes -0.5*log(det(M)) which is the log determinant term for a Normal N(0, M).
    """
    suffix = "next" if numerator_computation else "curr"
    match mass_params['strategy'], mass_params['matrix_type']:
        case "fixed", ("diag" | "full"):
            return 0  # numerator and denominator determinant terms would cancel out
        case ("schedule", ("diag" | "full")) | ("adaptive", "diag"):
            return -0.5*mass_params[f"log_det_mass_{suffix}"]
        case ("adaptive", "full"):
            raise NotImplemented("Mass Matrix adaptation with a full mass matrix is not implemented yet.")


def update_mass_matrix(mass_params: dict, xnk: NDArray, vnk: NDArray, nlps: NDArray, nlls: NDArray, gammas: list,
                       n: int, overflow_mask: NDArray) -> dict:
    """Selects the next mass matrix either using adaptation, a pre-defined schedule or keeping it fixed.

    Parameters
    ----------
    :param mass_params: Mass matrix parameters
    :type mass_params: dict
    :param xnk: Positions of shape (N, T+1, d)
    :type xnk: np.ndarray
    :param vnk: Velocities of shape (N, T+1, d)
    :type vnk: np.ndarray
    :param nlps: Negative log priors of shape (N, T+1)
    :type nlps: np.ndarray
    :param nlls: Negative log likelihoods of shape (N, T+1)
    :type nlls: np.ndarray
    :param gammas: Tempering parameters
    :type gammas: list
    :param n: Index of the current iteration should be between 1 and P
    :type n: int
    :param overflow_mask: Boolean mask of where computations for positions or velocities overflew
    :type overflow_mask: np.ndarray
    :return: Updated mass matrix parameters
    :rtype: dict
    """
    N, Tplus1, d = xnk.shape
    match mass_params["strategy"], mass_params["matrix_type"]:
        case "fixed", ("diag" | "full"):
            return mass_params  # do nothing
        case ("schedule" | "adaptive"), "diag":
            if mass_params['strategy'] == "schedule":
                mass_params['mass_diag_next'] = mass_params['schedule_func'](gammas[n])  # Mass Matrix diagonal
            else:
                W_mass_est, _, _, _, _ = compute_weights(
                    vnk=vnk, nlps=nlps, nlls=nlls, mass_params=mass_params, gamma_next=gammas[n], gamma_curr=gammas[n-1],
                    computation_for_mass_matrix_adaptation=True,
                    overflow_mask=overflow_mask
                )
                weighted_mean = np.average(xnk.reshape(-1, d)[~overflow_mask.ravel()], axis=0, weights=W_mass_est.ravel()[~overflow_mask.ravel()])  # (d, )
                mass_params['mass_diag_next'] = 1 / np.average((xnk.reshape(-1, d)[~overflow_mask.ravel()] - weighted_mean)**2, axis=0, weights=W_mass_est.ravel()[~overflow_mask.ravel()])
            mass_params['chol_mass_diag_next'] = np.sqrt(mass_params['mass_diag_next'])  # Cholesky(Mass Matrix)
            mass_params['log_det_mass_next'] = np.sum(np.log(mass_params['mass_diag_next']))  # Log determinant of the mass matrix
        case "schedule", "full":
            mass_params['mass_next'] = mass_params['schedule_func'](gammas[n])
            mass_params['chol_mass_next'] = np.linalg.cholesky(mass_params['mass_next'])
            mass_params['log_det_mass_next'] = np.linalg.slogdet(mass_params['mass_next']).logabsdet
        case "adaptive", "full":
            raise NotImplemented("Mass Matrix adaptation with a full mass matrix is not implemented yet.")
    return mass_params


def ess_from_log_weights(logw: NDArray) -> float:
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


def next_annealing_param(gamma: float, ESSrmin: float, llk: NDArray) -> float:
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


def leapfrog(x: NDArray, v: NDArray, T: int, epsilons: NDArray, gamma_curr: float, mass_params: dict,
             compute_likelihoods_priors_gradients: callable) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Leapfrog integration.

    Parameters
    ----------
    :param x: Positions of shape (N, d)
    :type x: np.ndarray
    :param v: Velocities of shape (N, d)
    :type v: np.ndarray
    :param T: Number of leapfrog steps
    :type T: int
    :param epsilons: Step sizes, one for each position-velocity pair
    :type epsilons: np.ndarray
    :param gamma_curr: Current tempering parameter, typically \\gamma_{n-1}
    :type gamma_curr: float
    :param mass_params: Parameters for the mass matrix
    :type mass_params: dict
    :param compute_likelihoods_priors_gradients: Function computing nlps, gnlps, nlls and gnlls
    :type compute_likelihoods_priors_gradients: callable
    :return: Positions (N, T+1, d), Velocities (N, T+1, d), Neg Log Priors (N, T+1), Neg Log Likelihood (N, T+1)
    :type: tuple
    """
    N, d = x.shape
    if len(epsilons.shape) == 1:
        epsilons = epsilons[:, None]

    # Select momentum update function
    def momentum_update_when_gamma_is_zero(gnlp, gnll):
        """Computes momentum update when gamma=0, meaning gradient of negative log likelihood is not used."""
        return epsilons * gnlp.reshape(gnll.shape)  # reshape used only for pylint

    def momentum_update_when_gamma_larger_than_zero(gnlp, gnll):
        """Computes momentum update when gamma>0, meaning gradient of negative log likelihood is used."""
        return epsilons*(gnlp + gamma_curr*gnll)
    mom_update = momentum_update_when_gamma_is_zero if gamma_curr == 0 else momentum_update_when_gamma_larger_than_zero

    # Initialize snippets, negative log priors and negative log likelihoods
    xnk = np.full((N, T+1, d), np.nan)
    vnk = np.full((N, T+1, d), np.nan)
    nlps = np.full((N, T+1), np.nan)  # negative log priors
    nlls = np.full((N, T+1), np.nan)  # negative log likelihoods
    xnk[:, 0] = x  # seed positions
    vnk[:, 0] = v  # seed velocities

    # First momentum half-step
    nlps[:, 0], gnlps, nlls[:, 0], gnlls = compute_likelihoods_priors_gradients(x)
    v = v - 0.5*mom_update(gnlp=gnlps, gnll=gnlls)  # (N, d)

    # T - 1 position and velocity full steps
    for k in range(T-1):

        # Full position step
        x = x + epsilons*inv_mass_times_v(v, mass_params)

        # Full momentum step
        nlps[:, k+1], gnlps, nlls[:, k+1], gnlls = compute_likelihoods_priors_gradients(x)
        v = v - mom_update(gnlp=gnlps, gnll=gnlls)

        # Store positions and velocities
        vnk[:, k+1] = v
        xnk[:, k+1] = x

    # Final position half-step
    x = x + epsilons*inv_mass_times_v(v, mass_params)

    # Final momentum half-step
    nlps[:, -1], gnlps, nlls[:, -1], gnlls = compute_likelihoods_priors_gradients(x)
    v = v - 0.5*mom_update(gnlp=gnlps, gnll=gnlls)

    # Store final position and velocity
    xnk[:, -1] = x
    vnk[:, -1] = v

    return xnk, vnk, nlps, nlls


def inv_mass_times_v(v: NDArray, mass_params: dict):
    """Computes M^{-1} v for the various scenarios.

    Parameters
    ----------
    :param v: Velocities to multiply by the inverse mass matrix
    :type v: np.ndarray
    :param mass_params: Parameters for the mass matrix
    :type mass_params: dict
    :return: Inverse mass matrix times v, an array of shape (N, d)
    :rtype: np.ndarray
    """
    if mass_params['matrix_type'] == "diag":
        return v / mass_params['mass_diag_curr'][None, :]  # (N, d)
    else:
        return np.linalg.solve(mass_params['mass_curr'], v.T).T  # (N, d)


def sample_epsilons(eps_params: dict, N: int, rng: np.random.Generator) -> NDArray:
    """Samples epsilons according to the correct distribution. Currently, only two options are available, either an
    inverse gaussian with a fixed skewness or a discrete uniform distribution.

    Parameters
    ----------
    :param eps_params: Step size parameters
    :type eps_params: dict
    :param N: Number of step sizes to sample, should be the same as the number of particles
    :type N: int
    :param rng: Random number generator for reproducibility
    :type: np.random.Generator
    :return: Array of sampled step sizes of shape (N, )
    :rtype: np.ndarray
    """
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
    Q_vals_history = []

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
            eps_params_update, Q_vals = estimate_with_cond_variance(
                xnk=xnk_for_eps_adaptation, logw_criterion=logw_criterion, epsilons=epsilons, epsilon_params=epsilon_params,
                skip_overflown=skip_overflown, overflow_mask=overflow_mask.any(axis=1), plot_Q_criterion=plot_Q_criterion
            )
            epsilon_params.update(eps_params_update)
            Q_vals_history.append(Q_vals)

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
            'coupling_success_history': coupling_success_history, 'Q_vals_history': Q_vals_history}


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


if __name__ == "__main__":
    # Grab data
    data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonar.npy"))
    y = -data[:, 0]  # Shape (208,)
    Z = data[:, 1:]  # Shape (208, 61)
    d = Z.shape[1]
    scales = np.array([5] * d)
    scales[0] = 20

    # Define function to sample the prior
    sample_prior = generate_sample_prior_function(_scales=scales)
    compute_likelihoods_priors_gradients = generate_nlp_gnlp_nll_and_gnll_function(_y=y, _Z=Z, _scales=scales)

    # Run Hamiltonian snippets
    n_runs = 1 # 20
    overall_seed = np.random.randint(low=0, high=10000000000)
    seeds = np.random.default_rng(overall_seed).integers(low=1, high=10000000000, size=n_runs)
    step_sizes = [0.3]  # np.array(np.geomspace(start=0.001, stop=10.0, num=9))  # np.array() used only for pylint
    N = 2500
    T = 30
    skewness = 8  # a large skewness helps avoiding a large bias
    verbose = True
    step_size_adaptation = True
    mass_params = {
        'strategy': 'fixed',
        'matrix_type': 'diag',
        'mass': np.ones(d),
    }

    results = []
    for i in range(n_runs):
        print(f"Run: {i}")
        for eps_ix, eps in enumerate(step_sizes):
            # epsilon_params = {
            #     'distribution': 'inv_gauss',
            #     'skewness': skewness,
            #     'mean': eps,
            #     'params_to_estimate': {'mean': lambda epsilon: epsilon},
            #     'to_print': 'mean'
            # }
            epsilon_params = {
                'distribution': 'discrete_uniform',
                'values': np.geomspace(start=0.001, stop=10.0, num=18),
                'params_to_estimate': {'values': lambda epsilon: epsilon},
                'to_print': 'values'
            }
            res = {'N': N, 'T': T, 'epsilon_params': {key: value for key, value in epsilon_params.items() if key != 'params_to_estimate'}}
            out = hamiltonian_snippet(N=N, T=T, ESSrmin=0.8,
                                      sample_prior=sample_prior,
                                      epsilon_params=epsilon_params,
                                      mass_params=mass_params,
                                      act_on_overflow=False,
                                      skip_overflown=False,
                                      compute_likelihoods_priors_gradients=compute_likelihoods_priors_gradients,
                                      adapt_step_size=step_size_adaptation,
                                      adapt_n_leapfrog_steps=False,
                                      plot_Q_criterion=True,
                                      plot_contractivity=False,
                                      max_contractivity=3,
                                      verbose=verbose, seed=seeds[i])
            res.update({'logLt': out['logLt'], 'out': out})
            print(f"\t\tEps: {eps: .7f} \tLogLt: {out['logLt']: .1f} \tFinal ESS: {out['ess'][-1]: .1f}"
                  f"\tEps {epsilon_params['to_print'].capitalize()}: "
                  f"Seed {int(seeds[i])} ")
            results.append(res)

    # Save results
    # with open(f"results_storage/Qvals_seed{overall_seed}_N{N}_T{T}_runs{n_runs}.pkl", "wb") as file:
    #     pickle.dump(results, file)
