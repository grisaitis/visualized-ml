import jax

from graphical_models.gaussian_mixture import GaussianMixture

locs = (15, -3)
scales = (0.1, 0.1)
mixture_weights = (0.3, 0.7)
key = jax.random.PRNGKey(seed=1)

gm = GaussianMixture(locs, scales, mixture_weights)

x = gm.sample(key, n=30)

print(x)

# gm2 = GaussianMixture.learn_from(x, k_mixtures=2)
# print(gm2.params)
# l = gm2.likelihood(x)
# print(l)
# ll = gm2.log_likelihood(x)
# print(ll)
