import jax


class GaussianMixture:
    def __init__(self, locs, scales, mixture_weights):
        self.locs = jax.numpy.array(locs)
        self.scales = jax.numpy.array(scales)
        self.mixture_weights = jax.numpy.array(mixture_weights)
        self.mixture_logits = jax.scipy.special.logit(mixture_weights)

    @staticmethod
    @jax.jit
    def sample_from_gaussian_mixture(key, locs, scales, mixture_logits, n):
        # print(n)
        n = int(n)
        # import sys; sys.exit()
        mixture_indices = jax.random.categorical(
            key, mixture_logits, shape=(int(n),)
        )
        gaussians = jax.random.normal(key, shape=(n,))
        return gaussians * scales.take(mixture_indices) + locs.take(
            mixture_indices
        )

    def sample_jit(self, n):
        # fails because of n being cast to a ScaledArray thing
        key = jax.random.PRNGKey(seed=0)
        # n = jax.numpy.array(n)
        return self.sample_from_gaussian_mixture(
            key, self.locs, self.scales, self.mixture_logits, n
        )

    def sample(self, key, n):
        mixture_indices = jax.random.categorical(
            key, self.mixture_logits, shape=(n,)
        )
        standard_normal_samples = jax.random.normal(key, shape=(n,))
        scales = self.scales.take(mixture_indices)
        locs = self.locs.take(mixture_indices)
        return standard_normal_samples * scales + locs


def _sample(key, locs, scales, size):
    return
