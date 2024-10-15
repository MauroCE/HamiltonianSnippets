import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from .leapfrog_integration import leapfrog
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'STIXGeneral'})


# def adapt_with_quantile(xnk, vnk, epsilons, nlps, nlls, T, gamma, inv_mass_diag, compute_likelihoods_priors_gradients, rng):
#     """Uses epsilons from 0.45 to 0.55 quantile of the epsilon distribution."""
#     # Select bulk of epsilons
#     N, Tplus1, d = xnk.shape
#     q_left, q_right = np.quantile(epsilons, q=[0.45, 0.55])
#     flag = (q_left <= epsilons) & (epsilons <= q_right)  # (N, ) indicates which epsilons to use
#
#     # Create coupled velocities and coupled epsilons
#     v_coupled, eps_coupled = vnk[flag, 0], epsilons[flag]  # (N_eps_flag, d), (N_eps_flag)
#     coupling = rng.permutation(x=flag.sum()).reshape(-1, 2)  # (N_eps_flag//2, )
#     v_coupled[coupling[:, 1]] = v_coupled[coupling[:, 0]]  # (N_eps_flag, d)
#     eps_coupled[coupling[:, 1]] = eps_coupled[coupling[:, 0]]  # (N_eps_flag, )
#
#     # Construct trajectories
#     xnk_coupled = xnk[flag]  # (N_eps_flag, T+1, d)
#     vnk_coupled = vnk[flag]  # (N_eps_flag, T+1, d)
#     nlps_coupled = nlps[flag]
#     nlls_coupled = nlls[flag]
#     xnk_coupled[coupling[:, 1]], vnk_coupled[coupling[:, 1]], nlps_coupled[coupling[:, 1]], nlls_coupled[coupling[:, 1]] = leapfrog(
#         x=xnk_coupled[coupling[:, 1], 0],
#         v=vnk_coupled[coupling[:, 1], 0],
#         T=T,
#         epsilons=eps_coupled[coupling[:, 1]],
#         gamma_curr=gamma,
#         inv_mass_diag_curr=inv_mass_diag,
#         compute_likelihoods_priors_gradients=compute_likelihoods_priors_gradients
#     )
#
#     # Compute squared distances between coupled trajectories
#     sq_norms = np.linalg.norm(xnk_coupled[coupling[:, 0]] - xnk_coupled[coupling[:, 1]], axis=2)**2  # (N_eps_flag//2, T+1)
#
#     # Compute integration times
#     taus = np.arange(Tplus1).reshape(1, -1) * eps_coupled[coupling[:, 0]].reshape(-1, 1)  # (N_eps_flag//2, T+1)
#     binned_avg_sq_norms, tau_bins = binned_average(sq_norms=sq_norms, taus=taus, n_bins=Tplus1)
#
#     fig, ax = plt.subplots()
#     ax.plot(taus.T, sq_norms.T, color='lightcoral', alpha=0.5)
#     ax.plot(tau_bins, binned_avg_sq_norms, lw=3, color='black')
#     ax.set_title(f"{taus.shape}, {sq_norms.shape}")
#     plt.show()
#
#     return T
#
#
# def adapt_with_mean_eps(xnk, vnk, eps_params, nlps, nlls, T, gamma, inv_mass_diag, compute_likelihoods_priors_gradients,
#                         overflow_mask, rng):
#     """Uses a single epsilon to perform the adaptation. Basically epsilons are not coupled because they are all
#     the same. """
#     # TODO: Deal with zero squared norms (probably due to overflow or something)
#     ofm = overflow_mask.any(axis=1)  # (N, ) indicates where overflow happened
#     N_ok, Tplus1, d = xnk[~ofm].shape
#
#     epsilons = np.repeat(eps_params[eps_params['param_for_T_adaptation']], N_ok)
#
#     # Couple velocities
#     v_coupled = vnk[~ofm, 0]  # (N_ok, d)
#     coupling = rng.permutation(x=N_ok).reshape(-1, 2)
#     v_coupled[coupling[:, 1]] = v_coupled[coupling[:, 0]]  # vnk[~ofm][coupling[:, 0], 0]
#
#     # Construct trajectories using duplicated velocities but a single step size
#     xnk[coupling[:, 1]], vnk[coupling[:, 1]], nlps[coupling[:, 1]], nlls[coupling[:, 1]] = leapfrog(
#         x=xnk[~ofm][coupling[:, 1], 0],
#         v=v_coupled[~ofm][coupling[:, 1]],
#         T=T,
#         epsilons=epsilons[coupling[:, 1]],
#         gamma_curr=gamma,
#         inv_mass_diag_curr=inv_mass_diag,
#         compute_likelihoods_priors_gradients=compute_likelihoods_priors_gradients
#     )
#
#     # Compute squared distances between coupled trajectories
#     sq_norms = np.linalg.norm(xnk[~ofm][coupling[:, 0]] - xnk[~ofm][coupling[:, 1]], axis=2)**2  # (N//2, T+1)
#     sq_norms /= sq_norms[:, 0].reshape(-1, 1)
#     avg_sq_norms = sq_norms.mean(axis=0)  # (T+1, )
#
#     fig, ax = plt.subplots()
#     ax.plot(np.arange(T+1), sq_norms.T, color='lightcoral', alpha=0.5)
#     ax.plot(np.arange(T+1), avg_sq_norms, lw=3, color='black')
#     plt.show()
#
#     # Optimal T is found from avg square norms
#     return np.argmin(avg_sq_norms) + 10


def adapt_num_leapfrog_steps_contractivity(
        xnk: NDArray, vnk: NDArray, epsilons: NDArray, nlps: NDArray, nlls: NDArray, T: int,
        gamma: float, inv_mass_diag: NDArray, compute_likelihoods_priors_gradients: callable,
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
    :param inv_mass_diag: Diagonal of the inverse of the mass matrix
    :type inv_mass_diag: np.array
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
            inv_mass_diag_curr=inv_mass_diag,
            compute_likelihoods_priors_gradients=compute_likelihoods_priors_gradients
        )

        # Compute contractivity between coupled trajectories
        contractivity = np.linalg.norm(xnk[coupling[:, 0]] - xnk[coupling[:, 1]], axis=2)  # (N//2, T+1)
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
        avg_contraction, tau_bins = binned_average(contractivity, taus, Tplus1)
        tau_min_contraction = tau_bins[np.argmin(avg_contraction)]  # tau corresponding to the minimum average contraction
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


# def adapt_num_leapfrog_steps(xnk: NDArray, vnk: NDArray, epsilons: NDArray, nlps: NDArray, nlls: NDArray, T: int,
#                              gamma: float, inv_mass_diag: NDArray, compute_likelihoods_priors_gradients: callable,
#                              rng: np.random.Generator):
#     """Not sure yet."""
#     N, Tplus1, d = xnk.shape
#
#     # Couple velocities and epsilons
#     v_coupled, eps_coupled = vnk[:, 0], epsilons
#     coupling = rng.permutation(x=N).reshape(-1, 2)
#     v_coupled[coupling[:, 1]] = vnk[coupling[:, 0], 0]
#     eps_coupled[coupling[:, 1]] = epsilons[coupling[:, 0]]
#
#     # Construct trajectories using duplicated velocities and step sizes
#     xnk[coupling[:, 1]], vnk[coupling[:, 1]], nlps[coupling[:, 1]], nlls[coupling[:, 1]] = leapfrog(
#         x=xnk[coupling[:, 1], 0],
#         v=v_coupled[coupling[:, 1]],
#         T=T,
#         epsilons=eps_coupled[coupling[:, 1]],
#         gamma_curr=gamma,
#         inv_mass_diag_curr=inv_mass_diag,
#         compute_likelihoods_priors_gradients=compute_likelihoods_priors_gradients
#     )
#
#     # Compute squared distances between coupled trajectories
#     sq_norms = np.linalg.norm(xnk[coupling[:, 0]] - xnk[coupling[:, 1]], axis=2)**2  # (N//2, T+1)
#
#     # Compute integration times
#     taus = np.arange(Tplus1).reshape(1, -1) * eps_coupled[coupling[:, 0]].reshape(-1, 1)  # (N//2, T+1)
#     binned_avg_sq_norms, tau_bins = binned_average(sq_norms=sq_norms, taus=taus, n_bins=Tplus1)
#
#     fig, ax = plt.subplots()
#     ax.plot(taus.T, sq_norms.T, color='lightcoral', alpha=0.5)
#     ax.plot(tau_bins, binned_avg_sq_norms, lw=3, color='black')
#     plt.show()
#
#     # For each trajectory find k where square norm is smallest and multiply by step size to find taus
#     # taus = np.argmin(sq_norms, axis=1) * eps_coupled[coupling[:, 0]]  # (N//2, ) integration times
#
#     return T


def binned_average(y_values, x_values, n_bins) -> Tuple[NDArray, NDArray]:
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
    binned_means = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            binned_means[i] = np.mean(y_values[mask])
    return binned_means, x_bins
