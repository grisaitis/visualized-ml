import jax.random

import numpy as np
import scipy.stats as stats

from graphical_models.gaussian_mixture.gaussian_mixture import GaussianMixture

np_rng = np.random.default_rng(seed=0)

k = 2
alpha_dirichlet = np.ones(shape=(k,)) * 5
weights = np_rng.dirichlet(alpha=alpha_dirichlet)
locs = np_rng.normal(size=(k,)) * 5
scales = np.abs(np_rng.normal(size=(k,))) + 1
oracle = GaussianMixture(weights, locs, scales)
print(oracle)
key = jax.random.PRNGKey(seed=1)
x = oracle.sample(key, n=3)
print(x)


def update_em(x, weights, locs, scales):
    n = len(x)
    assert x.shape == (n,), x.shape
    k = len(weights)
    assert weights.shape == (k,), weights.shape
    assert locs.shape == (k,), locs.shape
    assert scales.shape == (k,), scales.shape

    prob_xn_given_zn = stats.norm.pdf(x[:, None], locs, scales)  # (n, k)
    assert prob_xn_given_zn.shape == (n, k)
    prob_xn = prob_xn_given_zn @ weights  # (n,)
    assert prob_xn.shape == (n,)
    a1 = prob_xn_given_zn / prob_xn[:, None]
    assert a1.shape == (n, k)
    prob_zn_given_xn = weights * a1  # (n, k)
    assert prob_zn_given_xn.shape == (n, k)
    responsibilities = prob_zn_given_xn  # (n, k)

    x  # (n, k)
    n_k = np.sum(responsibilities, axis=0)  # (k,)
    assert n_k.shape == (k,)

    weights_new = n_k / n

    weighted_sum_of_x = responsibilities.T @ x  # (k,)
    assert weighted_sum_of_x.shape == (k,)
    locs_new = weighted_sum_of_x / n_k  # (k,)
    assert locs_new.shape == (k,)

    x_minus_locs_new = x[:, None] - locs_new  # (n, k)
    assert x_minus_locs_new.shape == (n, k)
    square_of_x_minus_locs_new = x_minus_locs_new * x_minus_locs_new  # (n, k)
    assert square_of_x_minus_locs_new.shape == (n, k)
    scales_new = np.sqrt(np.sum(responsibilities * square_of_x_minus_locs_new, axis=0) / n_k)
    assert scales_new.shape == (k,)

    prob_xn_given_zn_and_new_params = stats.norm.pdf(x[:, None], locs_new, scales_new)
    assert prob_xn_given_zn_and_new_params.shape == (n, k)
    prob_xn_given_new_params =  prob_xn_given_zn_and_new_params @ weights_new
    assert prob_xn_given_new_params.shape == (n,)
    log_likelihood = np.sum(np.log(prob_xn_given_new_params))

    return log_likelihood, weights_new, locs_new, scales_new


def learn_em(x, k):
    # np = jax.numpy

    # initialize
    weights = np.ones(shape=(k,)) / k
    locs = x[:k]  # assuming x is shuffled
    scales = np.ones(shape=(k,))
    print("weights", np.array(weights))
    print("locs", np.array(locs))
    print("scales", np.array(scales))

    t = 0
    log_likelihood_old = 0
    while True:
        t += 1
        log_likelihood, weights, locs, scales = update_em(x, weights, locs, scales)
        print("-" * 80)
        print("iteration", t)
        print("oracle", oracle)
        print("weights", np.array(weights))
        print("locs", np.array(locs))
        print("scales", np.array(scales))
        print("log_likelihood", log_likelihood)
        if abs(log_likelihood - log_likelihood_old) < 0.001:
            break
        if log_likelihood < log_likelihood_old:
            raise ValueError("optimization issue; log_likelihood got worse")
        assert not np.isnan(log_likelihood)
        log_likelihood_old = log_likelihood
    return

# learn_em = jax.jit(learn_em, static_argnums=(1,))


learn_em(x, k)

# gm = learn_likelihood_only(x, 2)
# print(gm)

# gm2 = GaussianMixture.learn_from(x, k_mixtures=2)
# print(gm2.params)
# l = gm2.likelihood(x)
# print(l)
# ll = gm2.log_likelihood(x)
# print(ll)
