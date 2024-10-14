import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from .leapfrog_integration import leapfrog
import matplotlib.pyplot as plt


def adapt_num_leapfrog_steps(xnk: NDArray, vnk: NDArray, epsilons: NDArray, nlps: NDArray, nlls: NDArray, T: int,
                             gamma: float, inv_mass_diag: NDArray, compute_likelihoods_priors_gradients: callable,
                             rng: np.random.Generator):
    """Not sure yet."""
    N, Tplus1, d = xnk.shape

    # Couple velocities and epsilons
    v_coupled, eps_coupled = vnk[:, 0], epsilons
    coupling = rng.permutation(x=N).reshape(-1, 2)
    v_coupled[coupling[:, 1]] = vnk[coupling[:, 0], 0]
    eps_coupled[coupling[:, 1]] = epsilons[coupling[:, 0]]

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

    # Compute squared distances between coupled trajectories
    sq_norms = np.linalg.norm(xnk[coupling[:, 0]] - xnk[coupling[:, 1]], axis=2)**2  # (N//2, T+1)

    # Compute integration times
    taus = np.arange(Tplus1).reshape(1, -1) * eps_coupled[coupling[:, 0]].reshape(-1, 1)  # (N//2, T+1)
    binned_avg_tau, tau_bins = binned_average(sq_norms=sq_norms, taus=taus, n_bins=Tplus1)

    fig, ax = plt.subplots()
    ax.plot(taus.T, sq_norms.T, color='lightcoral', alpha=0.5)
    ax.plot(tau_bins, binned_avg_tau, lw=3, color='black')
    plt.show()

    # For each trajectory find k where square norm is smallest and multiply by step size to find taus
    # taus = np.argmin(sq_norms, axis=1) * eps_coupled[coupling[:, 0]]  # (N//2, ) integration times

    pass


def binned_average(sq_norms, taus, n_bins):
    """Computes a binned average."""
    bins = np.linspace(start=taus.min(), stop=taus.max(), num=n_bins)
    bin_indices = np.digitize(taus, bins, right=True)  # since taus will include zeros
    y_binned_means = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            y_binned_means[i] = np.mean(sq_norms[mask])
    return y_binned_means, bins
