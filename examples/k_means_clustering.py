import time

import jax
import numpy as np

from visualized_ml.non_probabilistic.k_means_mixture import KMeansMixture

rng = np.random.default_rng(seed=0)

n = 100
k = 10
scale = 0.1
locs = np.arange(n) % k + 1
x = locs + scale * rng.normal(size=(n,))
print(x)
print(x.shape)

key = jax.random.PRNGKey(seed=0)
start = time.time()
kmm = KMeansMixture.learn_with_lloyds_algorithm(key, x, k)
print("means (final)", kmm.means)
print("done in", time.time() - start, "seconds")
