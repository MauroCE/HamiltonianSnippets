import time
import numpy as np
from typing import Optional

from utils import next_annealing_param


def hamiltonian_snippet(N: int, ESSrmin: float, sample_prior: callable, verbose: bool = True, seed: Optional[int] = None):
    """Some"""
    # Set up time-keeping, random number generation and printing
    start_time = time.time()
    rng = np.random.default_rng(seed=seed if seed is not None else np.random.randint(low=0, high=10000000))
    verboseprint = print if verbose else lambda *a, **kwargs: None
    n = 1  # iteration number

    # Initialize particles
    x = sample_prior(N, rng)

    # Storage
    gammas = [0.0]

    while gammas[n-1] < 1.0:
        verboseprint(f"Iteration {n} Gamma {gammas[n-1]}")

        # Construct trajectories

        # Select next tempering parameter based on target ESS
        gammas.append(next_annealing_param(gamma=gammas[n-1], ESSrmin=ESSrmin, llk=(-nlls[:, 0])))

        # Compute unfolded weights and ESS

        # Resampling

        # Refresh velocities

        # Step size adaptation

        # Storage
        pass
    pass
