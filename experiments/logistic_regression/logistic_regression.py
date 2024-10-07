import numpy as np
from HamiltonianSnippets.sampler import hamiltonian_snippet
from HamiltonianSnippets.utils import eps_to_str
import pickle


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
    data = np.load("logistic_regression/sonar.npy")
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
    step_sizes = np.geomspace(start=0.001, stop=10.0, num=9)
    N = 500
    T = 30
    mass_matrix_adaptation = False
    mass_diag = 1 / scales**2 if mass_matrix_adaptation else np.ones(61)
    verbose = False

    results = []
    for i in range(n_runs):
        print(f"Run: {i}")
        for eps_ix, eps in enumerate(step_sizes):
            res = {'N': N, 'T': T, 'epsilon': eps}
            out = hamiltonian_snippet(N=N, T=T, step_size=eps, mass_diag=mass_diag, ESSrmin=0.8,
                                      sample_prior=sample_prior,
                                      compute_likelihoods_priors_gradients=compute_likelihoods_priors_gradients,
                                      adapt_mass=mass_matrix_adaptation,
                                      verbose=verbose, seed=seeds[i])
            res.update({'logLt': out['logLt'], 'out': out})
            print(f"\t\tEps: {eps: .7f} \tLogLt: {out['logLt']: .1f} \tFinal ESS: {out['ess'][-1]: .1f}"
                  f"\tEps Mean: {out['epsilon_means'][-1]: .3f} Seed {int(seeds[i])} ")
            results.append(res)

    # Save results
    with open(f"results/seed{overall_seed}_N{N}_T{T}_mass{mass_matrix_adaptation}_runs{n_runs}_from{eps_to_str(min(step_sizes))}_to{eps_to_str(max(step_sizes))}.pkl", "wb") as file:
        pickle.dump(results, file)
