import os
import numpy as np
from HamiltonianSnippets.sampler import hamiltonian_snippet
from HamiltonianSnippets.utils import eps_to_str
import pickle
import pandas as pd
import scipy as sp
from copy import deepcopy
import math


def generate_nlp_gnlp_nll_and_gnll_function(_y, _Z, _scales):
    """Computes negative log likelihood and its gradient manually.

    nll(x) = sum_{i=1}^{n_d} log(1 + exp(-y_i x^top z_i))

    gnll(x) = sum_{i=1}^{n_d} frac{exp(-y_i x^top z_i)}{1 + exp(-y_i x^top z_i)} y_i z_i
    """
    def nlp_gnlp_nll_and_gnll(x):
        nlp = np.full(fill_value=np.inf, shape=x.shape[0])  # (N, )
        nll = np.full(fill_value=np.inf, shape=x.shape[0])  # (N, )

        # Negative log prior
        nlp_flag = np.any(x < np.sqrt(np.finfo(np.float64).max) * _scales.max(), axis=1)
        # nlp[nlp_flag] = 0.5*61*np.log(2*np.pi) + 0.5*np.log(400.*(25.0**60)) + 0.5*np.sum((x[nlp_flag] / _scales)**2, axis=1)
        nlp[nlp_flag] = 0.5*61*np.log(2*np.pi) + 0.5*np.log(400) + 0.5*60.0*np.log(25.0) + 0.5*np.sum((x[nlp_flag] / _scales)**2, axis=1)
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

    n_runs = 20
    overall_seed = np.random.randint(low=0, high=10000000000)
    seeds = np.random.default_rng(overall_seed).integers(low=1, high=10000000000, size=n_runs)
    step_sizes = np.array(np.geomspace(start=0.001, stop=10.0, num=9, endpoint=True))  # np.array() used only for pylint

    # Settings
    N = 500
    T = 80
    skewness = 3
    aoo = False  # whether to act on overflow
    skipo = False  # skip overflown trajectories in epsilon computation
    verbose = False
    plot_contractivity = False
    plot_Q_criterion = False
    adapt_T = True
    adapt_epsilon = True
    adapt_mass = False
    save_contractivity_fig = False
    contractivity_save_path = None  # "results/contractivity_figures/"
    T_max = T
    T_min = 2
    max_contractivity = 1.5
    bottom_quantile_val = 0.05
    max_tries_find_coupling = 20
    mass_params = {
        'strategy': 'fixed',
        'matrix_type': 'diag',
        'mass': np.ones(d),
    }

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
                'on_overflow': lambda param_dict: {'skewness': max(1, param_dict['skewness'] * 0.99)},
                'param_for_T_adaptation': 'mean'
            }
            res = {'N': N, 'T': T, 'epsilon_params': {key: value for key, value in epsilon_params.items() if key not in ["params_to_estimate", "on_overflow", "mode_func"]}}
            try:
                out = hamiltonian_snippet(N=N, T=T, ESSrmin=0.8,
                                          sample_prior=sample_prior,
                                          compute_likelihoods_priors_gradients=compute_likelihoods_priors_gradients,
                                          epsilon_params=epsilon_params,
                                          mass_params=mass_params,
                                          act_on_overflow=aoo,
                                          skip_overflown=skipo,
                                          adapt_step_size=adapt_epsilon,
                                          adapt_n_leapfrog_steps=adapt_T,
                                          plot_contractivity=plot_contractivity,
                                          plot_Q_criterion=plot_Q_criterion,
                                          T_max=T_max,
                                          T_min=T_min,
                                          max_tries_find_coupling=max_tries_find_coupling,
                                          max_contractivity=max_contractivity,
                                          bottom_quantile_val=bottom_quantile_val,
                                          save_contractivity_fig=save_contractivity_fig,
                                          contractivity_save_path=contractivity_save_path,
                                          verbose=verbose,
                                          seed=seeds[i])
                print(f"\t\tEps: {eps: .7f} \tLogLt: {out['logLt']: .1f} \tFinal ESS: {out['ess'][-1]: .1f}"
                      f"\tEps {epsilon_params['to_print'].capitalize()}: {out['epsilon_params_history'][-1][epsilon_params['to_print']]} "
                      f"\tFinal T: {out['T_history'][-1]}"
                      f"\tSeed {int(seeds[i])}")
            except AssertionError:
                out = {"logLt": np.nan, "gammas": [], "runtime": np.nan, "epsilons": np.nan, "ess": np.nan,
                       'epsilon_params_history': np.nan, 'T_history': np.nan,
                       'coupling_success_history': np.nan}
                print(f"\t\tEps: {eps: .7f} \tLogLt: {np.nan} \tFinal ESS: {np.nan}"
                      f"\tEps: {np.nan} Final T: {np.nan}"
                      f"Seed {int(seeds[i])} ")
            res.update({'logLt': out['logLt'], 'out': out})
            results.append(res)

    with open(f"results/new_adaptT_sonar_seed{overall_seed}_N{N}_T{T}_massFalse_runs{n_runs}_from{eps_to_str(min(step_sizes))}_to{eps_to_str(max(step_sizes))}_skewness{skewness}_aoo{aoo}_skipo{skipo}_minT{T_min}.pkl", "wb") as file:
        pickle.dump(results, file)
