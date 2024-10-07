from scipy.optimize import brentq
import numpy as np


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


def eps_to_str(epsilon):
    """Simply replace . with 'dot'"""
    return str(float(epsilon)).replace(".", "dot")
