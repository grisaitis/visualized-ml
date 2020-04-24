import jax

from graphical_models.non_probabilistic.k_means_mixture import KMeansMixture

n = 1000000
key = jax.random.PRNGKey(seed=0)
k = 3

gaussian_samples = jax.random.normal(key, shape=(n,)) * 0.1
offsets = jax.numpy.arange(n) % 3
x = gaussian_samples + offsets * 5 - 5

print(x)

kmm = KMeansMixture.learn_with_lloyds_algorithm(key, k_means=k, x=x)

print(kmm.means)
