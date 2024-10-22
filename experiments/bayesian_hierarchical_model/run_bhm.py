import numpy as np
from HamiltonianSnippets.core.sampler import hamiltonian_snippet


def generate_data(data_dim: int, seed: int):
    """Generates data for the Bayesian Hierarchical Model in https://arxiv.org/pdf/2407.20722."""
    rng = np.random.default_rng(seed=seed)
    # Sample `dim` latent variables
    z = rng.normal(loc=0, scale=np.exp(-2.0), size=data_dim)  # (dim, )
    # Sample `dim` observed data
    return z + rng.normal(loc=0, scale=1.0, size=data_dim)  # (dim, )


def nlp_gnlp_nll_and_gnll(x):
    """Computes negative log prior, gradient negative log prior, negative log likelihood and gradient negative log-likelihood.
    Assumes tau=2, theta=-2 and sigma=1. The input is `x = (theta, z)` and therefore we expect it to have shape (N, dim+1)."""

    # Negative log prior
    nlp = 0.125 * x[:, 0]**2   # (N, ) -log_p(theta) assuming tau=2
    nlp += 0.5*np.einsum('ij,ij->i', x[:, 1:], x[:, 1:]) / np.exp(x[:, 0])  # (N, ) -log_p(z|theta)

    # Gradient negative log prior
    gnlp = np.full(fill_value=np.inf, shape=x.shape)
    gnlp[:, 0] = 0.5*x[:, 0]  # (N, 1) - nabla_log_p(theta) assuming tau=2
    gnlp[:, 1:] = x[:, 1:] / np.exp(x[:, 0])  # (N, dim) - nala_log_p(z|theta)


if __name__ == "__main__":
    # Generate data
    dim = 30
    data_seed = 1234
    data = generate_data(data_dim=dim, seed=data_seed)

    # We consider (z, theta) to be our parameter, which will be `(dim+1)`-dimensional. This means that our prior
    # corresponds to `p(z, theta)` and our likelihood to `p(D | z, theta) = p(D | z)`, i.e. it does not depend on theta.



    # Define function to sample the prior
    sample_prior = generate_sample_prior_function()
    compute_likelihoods_priors_gradients = generate_nlp_gnlp_nll_and_gnll_function()

    # Run Hamiltonian snippets
    n_runs = 20
    overall_seed = np.random.randint(low=0, high=10000000000)
    seeds = np.random.default_rng(overall_seed).integers(low=1, high=10000000000, size=n_runs)
    step_sizes = [0.17]  # np.array(np.geomspace(start=0.001, stop=10.0, num=9))  # np.array() used only for pylint
    N = 1000
    T = 50
    skewness = 1  # a large skewness helps avoiding a large bias
    mass_matrix_adaptation = False
    mass_diag = np.ones(61)
    verbose = False
    step_size_adaptation = True
    T_adaptation = True

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
                'param_for_T_adaptation': 'mean'
            }
            res = {'N': N, 'T': T, 'epsilon_params': epsilon_params}
            out = hamiltonian_snippet(N=N, T=T, mass_diag=mass_diag, ESSrmin=0.8,
                                      sample_prior=sample_prior,
                                      epsilon_params=epsilon_params,
                                      act_on_overflow=False,
                                      skip_overflown=False,
                                      compute_likelihoods_priors_gradients=compute_likelihoods_priors_gradients,
                                      adapt_mass=mass_matrix_adaptation,
                                      adapt_step_size=step_size_adaptation,
                                      adapt_n_leapfrog_steps=T_adaptation,
                                      verbose=verbose, seed=seeds[i])
            res.update({'logLt': out['logLt'], 'out': out})
            print(f"\t\tEps: {eps: .7f} \tLogLt: {out['logLt']: .1f} \tFinal ESS: {out['ess'][-1]: .1f}"
                  f"\tEps {epsilon_params['to_print'].capitalize()}: "
                  f"{out['epsilon_params_history'][-1][epsilon_params['to_print']]: .3f} Seed {int(seeds[i])} ")
            results.append(res)
