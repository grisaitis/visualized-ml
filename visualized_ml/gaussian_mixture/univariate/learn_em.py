import logging
import time

import numpy as np
import scipy.stats as stats

from .univariate_gaussian_mixture import UnivariateGaussianMixture

np_rng = np.random.default_rng(seed=0)
logger = logging.getLogger(__name__)


def compute_prob_zi_given_xi(x, weights, locs, scales):
    n = len(x)
    k = len(weights)
    prob_xi_given_zi = stats.norm.pdf(
        x[:, None], loc=locs, scale=scales
    )  # (n, k)
    assert prob_xi_given_zi.shape == (n, k)
    prob_xi = prob_xi_given_zi @ weights  # (n,)
    assert prob_xi.shape == (n,)
    a1 = prob_xi_given_zi / prob_xi[:, None]
    assert a1.shape == (n, k)
    prob_zi_given_xi = weights * a1  # (n, k)
    assert prob_zi_given_xi.shape == (n, k)
    return prob_zi_given_xi


def update_em(x, weights, locs, scales):
    n = len(x)
    k = len(weights)
    assert x.shape == (n,), x.shape
    assert weights.shape == (k,), weights.shape
    assert locs.shape == (k,), locs.shape
    assert scales.shape == (k,), scales.shape

    prob_zi_given_xi = compute_prob_zi_given_xi(x, weights, locs, scales)

    n_k = np.sum(prob_zi_given_xi, axis=0)  # (k,)
    assert n_k.shape == (k,)
    # print("n_k", n_k)

    weights_new = n_k / np.sum(n_k)
    weights_new = weights_new / np.sum(weights_new)
    # print("weights_new", weights_new, np.sum(weights_new), weights_new.dtype)
    assert np.isclose(np.sum(weights_new), 1.0), (
        weights_new,
        np.sum(weights_new),
    )

    # print("x * prob_zi_given_xi\n", x[:, None] * prob_zi_given_xi)
    weighted_sum_of_x = prob_zi_given_xi.T @ x  # (k,)
    assert weighted_sum_of_x.shape == (k,)
    # print("weighted_sum_of_x", weighted_sum_of_x)
    locs_new = weighted_sum_of_x / n_k  # (k,)
    # print("locs_new", locs_new)
    assert locs_new.shape == (k,)

    x_minus_locs_new = x[:, None] - locs_new  # (n, k)
    assert x_minus_locs_new.shape == (n, k)
    square_of_x_minus_locs_new = x_minus_locs_new * x_minus_locs_new  # (n, k)
    assert square_of_x_minus_locs_new.shape == (n, k)
    scales_new = np.sqrt(
        np.sum(prob_zi_given_xi * square_of_x_minus_locs_new, axis=0) / n_k
    )
    assert scales_new.shape == (k,)

    return weights_new, locs_new, scales_new


def initialize_parameters(k):
    weights = np.ones(shape=(k,)) / k
    locs = np_rng.normal(size=(k,))
    scales = np.ones(shape=(k,))  # * (2 * 3.14159) ** (-0.5)
    return weights, locs, scales


def learn_em(x, k, oracle=None, max_iter=1e10):
    weights, locs, scales = initialize_parameters(k)
    gaussian_mixture = UnivariateGaussianMixture(weights, locs, scales)

    t = 0
    average_log_likelihood_old = gaussian_mixture.log_likelihood(x) / len(x)
    while True:
        t += 1
        weights, locs, scales = update_em(x, weights, locs, scales)
        gaussian_mixture = UnivariateGaussianMixture(weights, locs, scales)
        average_log_likelihood = gaussian_mixture.log_likelihood(x) / len(x)
        improvement = average_log_likelihood - average_log_likelihood_old
        is_done = (improvement < 1e-8) or (max_iter <= t)
        logger.debug(f"iteration={t}, improvement={improvement:e}", extra={"data": {
            "iteration": t,
            "gaussian_mixture": gaussian_mixture,
            "oracle": oracle,
            "average_log_likelihood": average_log_likelihood,
            "improvement": improvement,
            "is_done": is_done,
        }})
        assert not np.isnan(average_log_likelihood)
        if improvement < 0:
            raise ValueError("optimization error: likelihood got worse")
        if is_done:
            break
        average_log_likelihood_old = average_log_likelihood
    return UnivariateGaussianMixture(weights, locs, scales)
