import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from .leapfrog_integration import leapfrog
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'STIXGeneral'})


def adapt_num_leapfrog_steps_contractivity(
        xnk: NDArray, vnk: NDArray, epsilons: NDArray, nlps: NDArray, nlls: NDArray, T: int,
        gamma: float, mass_params: dict, compute_likelihoods_priors_gradients: callable,
        rng: np.random.Generator, max_tries: int = 100, plot_contractivity: bool = False, T_max: int = 100,
        T_min: int = 5, max_contractivity: float = 3, bottom_quantile_val: float = 0.05,
        save_contractivity_fig: bool = False, contractivity_save_path: Optional[str] = None,
        n: Optional[int] = None, seed: Optional[int] = None) -> Tuple[int, bool]:
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
    :param max_contractivity: Maximum contractivity that contractivities are clipped to
    :type max_contractivity: float
    :param bottom_quantile_val: Quantile value for the minimum contractivity
    :type bottom_quantile_val: float
    :param save_contractivity_fig: Whether to save the contractivity figures at each step
    :type save_contractivity_fig: bool
    :param contractivity_save_path: Path where to save the contractivity figures, defaults to None
    :type contractivity_save_path: str
    :param n: Iteration number
    :type n: int
    :param seed: Reproducibility seed, used only for saving pictures
    :type seed: int
    :return: A tuple (T, coupling_found) where T is the optimal num of leapfrog steps and the other a boolean flag
    :rtype: tuple
    """
    assert isinstance(max_tries, int) and max_tries >= 1, "Maximum number of tries should be an integer >= 1."
    assert isinstance(T_min, int) and T_min >= 1, "Minimum number of integration steps must be an integer >= 1."
    assert isinstance(max_contractivity, int | float) and max_contractivity >= 1, "Max contractivity must be at least 1."
    assert 0 <= bottom_quantile_val <= 1.0, "bottom_quantile_val must be in [0,1]."
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
        contractivity = np.clip(contractivity, a_min=None, a_max=max_contractivity)  # to avoid coupled particles diverging too much

        # Compute a binned average of the contractivity as a function of tau
        n_bins = max(100, np.sqrt(N))
        taus = np.arange(Tplus1).reshape(1, -1) * eps_coupled[coupling[:, 0]].reshape(-1, 1)  # (N//2, T+1) Integration times corresponding to coupled particles
        avg_contraction, tau_bins = binned_average(contractivity, taus, n_bins)  # (n_bins, ), (n_bins, )
        argmin_avg_contraction = np.nanargmin(avg_contraction)
        tau_min_contraction = 0.5*(tau_bins[argmin_avg_contraction - 1] + tau_bins[argmin_avg_contraction])  # tau corresponding to the minimum average contraction

        # Compute the new T as the tau corresponding to the average min contraction divided by the median step size
        # among particles in the tail of the contraction distribution
        denominator = np.quantile(epsilons, q=0.5)
        T_optimal = np.ceil(tau_min_contraction/denominator).astype(int)

        if plot_contractivity:
            fig, ax = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
            # Subplot 1: contractivity as a function of tau
            ax[0].plot(taus.T, contractivity.T, color='lightcoral', alpha=0.5)
            ax[0].plot(tau_bins, avg_contraction, color='black', label='Binned Average', lw=1)
            ax[0].axvline(tau_min_contraction, color='dodgerblue', ls='--', label=r'$\mathregular{\tau_{n-1}^*}$')
            ax[0].set_xlabel(r"$\mathregular{\tau}$", fontsize=13)
            ax[0].set_ylabel("Contractivity", fontsize=13)
            ax[0].legend()
            ax[0].grid(True, color='gainsboro')
            # Subplot 2: 2D Histogram of contractivity curves
            ax[1].hist2d(taus.ravel(), contractivity.ravel(), bins=100)
            ax[1].set_xlabel(r"$\mathregular{\tau}$", fontsize=13)
            if save_contractivity_fig:
                plt.savefig(contractivity_save_path + f"{n}_{seed}.png")
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
