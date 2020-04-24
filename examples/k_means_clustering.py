import time

import jax

from graphical_models.non_probabilistic.k_means_mixture import KMeansMixture

n = 1000000
key = jax.random.PRNGKey(seed=0)
k = 10
gaussian_samples = jax.random.normal(key, shape=(n,))
offsets = jax.numpy.arange(n) % k
x = gaussian_samples + offsets + 1
print(x)
start = time.time()
kmm = KMeansMixture.learn_with_lloyds_algorithm(key, x, k)
print("means (final)", kmm.means)
print("done in", time.time() - start, "seconds")
