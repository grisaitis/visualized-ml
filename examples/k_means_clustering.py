import jax

from graphical_models.non_probabilistic.k_means_mixture import KMeansMixture

n = 1000000
key = jax.random.PRNGKey(seed=0)

gaussian_samples = jax.random.normal(key, shape=(n,))
x = jax.numpy.concatenate(
    (
        gaussian_samples[: int(n / 3)] - 5,
        gaussian_samples[int(n / 3) : n - int(n / 3)],
        gaussian_samples[n - int(n / 3) :] + 5,
    )
)

# print(x)

kmm = KMeansMixture.learn_from_data(key, k_means=3, x=x)

print(kmm.means)
