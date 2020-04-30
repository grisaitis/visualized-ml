import numpy as np

from graphical_models.gaussian_mixture import (
    GaussianMixture,
    learn_likelihood_gradient,
)

np.set_printoptions(suppress=True)
np_rng = np.random.default_rng(seed=0)

k = 2
alpha_dirichlet = np.ones(shape=(k,)) * 5
weights = np_rng.dirichlet(alpha=alpha_dirichlet)
locs = np_rng.normal(size=(k,)) * 5
scales = np.abs(np_rng.normal(size=(k,)))
oracle = GaussianMixture(weights, locs, scales)

x = oracle.sample(seed=1, n=100_000)

gmm_learned = learn_likelihood_gradient(x, k, oracle)

print("done. learned gaussian mixture:")
print(gmm_learned)

print("and the 'oracle' (truth) was:")
print(oracle)
