import jax
import numpy as np
import scipy.stats as stats

from graphical_models.gaussian_mixture.gaussian_mixture import GaussianMixture

np.set_printoptions(suppress=True)

np_rng = np.random.default_rng(seed=0)

# weights = np.array([0.4, 0.6])
k = 4
alpha = np.ones((k,)) * 10
weights = np_rng.dirichlet(alpha)
# locs = np.array([-5.0, 5.0])
locs = np_rng.normal(size=(k,)) * 5
# scales = np.array([1.0, 2.0])
scales = np_rng.normal(size=(k,)) * 0.1 + 1

print(weights)
print(locs)
print(scales)

import time

time.sleep(2)

oracle = GaussianMixture(weights, locs, scales)
print(oracle)
x = oracle.sample(seed=1, n=10000)
print(x)


def log_likelihood(x, weights, locs, scales):
    np = jax.numpy
    stats = jax.scipy.stats
    n = len(x)
    k = len(locs)
    probs = stats.norm.pdf(np.expand_dims(x, axis=-1), loc=locs, scale=scales)
    assert probs.shape == (n, k)
    # weights = np.concatenate((weights[:-1], np.array([1.0 - np.sum(weights[:-1])])))
    assert weights.shape == (k,)
    assert np.isclose(np.sum(weights), 1.0), (weights, np.sum(weights))
    log_likelihoods = np.log(np.dot(probs, weights))  # (n, k) x (k,)
    assert log_likelihoods.shape == (n,)
    return np.mean(log_likelihoods)


ll = log_likelihood(x, weights, locs, scales)
print(ll)


def initialize_parameters(k):
    weights = np.ones(shape=(k,)) / k
    locs = np_rng.normal(size=(k,))
    scales = np.ones(shape=(k,))  # * (2 * 3.14159) ** (-0.5)
    return weights, locs, scales


def learn_likelihood_only(x, k):
    weights, locs, scales = initialize_parameters(k)

    t = 0
    step = 0.1
    ll_value_old = -99999999999
    while True:
        t += 1
        weights = weights / np.sum(weights)
        log_likelihood_grad = jax.value_and_grad(
            log_likelihood, argnums=(1, 2, 3)
        )
        ll_value, grads = log_likelihood_grad(x, weights, locs, scales)
        weights_grad, locs_grad, scales_grad = grads
        weights = weights + step * weights_grad
        weights = np.clip(weights, 0, 1)
        weights = weights / np.sum(weights)
        locs = locs + step * locs_grad
        scales = scales + step * scales_grad
        print("iteration", t)
        print("oracle", oracle)
        print("log_likelihood", ll_value)
        print("weights and grad\n", np.array([weights, weights_grad]))
        print("locs and grad\n", np.array([locs, locs_grad]))
        print("scales and grad\n", np.array([scales, scales_grad]))
        print("learned", GaussianMixture(weights, locs, scales))
        if np.any(weights < 0):
            break
        if abs(ll_value - ll_value_old) < 0.0000000001:
            break
        ll_value_old = ll_value
    return


learn_likelihood_only(x, k)
