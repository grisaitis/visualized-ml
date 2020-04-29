import jax


class GaussianMixture:
    def __init__(self, weights, locs, scales):
        self.k = len(weights)
        assert weights.shape == (self.k,)
        assert locs.shape == (self.k,)
        assert scales.shape == (self.k,)
        assert sum(weights) == 1
        self.weights = weights
        self.locs = locs
        self.scales = scales

    def __repr__(self):
        return (
            f"GaussianMixture("
            f"\n\tweights={repr(self.weights)},"
            f"\n\tlocs={repr(self.locs)},"
            f"\n\tscales={repr(self.scales)}"
            f"\n)"
        )

    def sample(self, key, n):
        return _sample(key, n, self.weights, self.locs, self.scales)


def _sample(key, n, weights, locs, scales):
    logits = jax.scipy.special.logit(weights)
    mixture_indices = jax.random.categorical(key, logits, shape=(n,))
    sample_scales = scales.take(mixture_indices)
    sample_locs = locs.take(mixture_indices)
    normals = jax.random.normal(key, shape=(n,))
    return normals * sample_scales + sample_locs


_sample = jax.jit(_sample, static_argnums=(1,))
