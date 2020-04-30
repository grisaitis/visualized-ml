import jax
import numpy as np
import scipy.stats as stats

from . import GaussianMixture

np_rng = np.random.default_rng(seed=0)


@jax.jit
def log_likelihood_differentiable(x, weights, locs, scales):
    np = jax.numpy
    stats = jax.scipy.stats
    n = len(x)
    k = len(locs)
    probs = stats.norm.pdf(np.expand_dims(x, axis=-1), loc=locs, scale=scales)
    assert probs.shape == (n, k)
    # weights = np.concatenate((weights[:-1], np.array([1.0 - np.sum(weights[:-1])])))
    assert weights.shape == (k,)
    # assert np.isclose(np.sum(weights), 1.0), (weights, np.sum(weights))
    log_likelihoods = np.log(np.dot(probs, weights))  # (n, k) x (k,)
    # assert log_likelihoods.shape == (n,)
    return np.mean(log_likelihoods)


log_likelihood_grad = jax.grad(log_likelihood_differentiable, argnums=(1, 2, 3))


def initialize_parameters(k):
    weights = np.ones(shape=(k,)) / k
    locs = np_rng.normal(size=(k,))
    scales = np.ones(shape=(k,))  # * (2 * 3.14159) ** (-0.5)
    return weights, locs, scales


def learn_likelihood_gradient(x, k, oracle=None, step=0.1):
    weights, locs, scales = initialize_parameters(k)

    t = 0
    log_likelihood_old = GaussianMixture(
        weights, locs, scales
    ).compute_log_likelihood(x)
    while True:
        t += 1
        weights_grad, locs_grad, scales_grad = log_likelihood_grad(
            x, weights, locs, scales
        )
        # project grad vector into probability simplex
        weights_grad -= np.mean(weights_grad)
        # update parameters
        weights = weights + step * weights_grad
        weights = np.clip(weights, 1e-8, 1 - 1e-8)
        weights = weights / np.sum(weights)
        locs = locs + step * locs_grad
        scales = scales + step * scales_grad
        gmm_learned = GaussianMixture(weights, locs, scales)
        log_likelihood = gmm_learned.compute_log_likelihood(x)
        print("-" * 80)
        print("iteration", t)
        if oracle:
            print("oracle", oracle)
        print("learned", gmm_learned)
        print("log_likelihood", log_likelihood)
        print(
            "improvement", log_likelihood - log_likelihood_old,
        )
        assert not np.isnan(log_likelihood)
        if np.any(weights < 0):
            break
        if log_likelihood < log_likelihood_old:
            raise ValueError("optimization issue; log_likelihood got worse")
        if abs(log_likelihood - log_likelihood_old) < 1e-8:
            break
        log_likelihood_old = log_likelihood
    print("-" * 80)
    return GaussianMixture(weights, locs, scales)
