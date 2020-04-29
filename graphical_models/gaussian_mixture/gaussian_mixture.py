import numpy as np


class GaussianMixture:
    def __init__(self, weights, locs, scales):
        self.k = len(weights)
        assert weights.shape == (self.k,)
        assert locs.shape == (self.k,)
        assert scales.shape == (self.k,)
        assert np.isclose(np.sum(weights), 1.0), (weights, np.sum(weights))
        sort_order = locs.argsort()
        self.weights = weights.take(sort_order)
        self.locs = locs.take(sort_order)
        self.scales = scales.take(sort_order)

    def __repr__(self):
        return (
            f"GaussianMixture("
            f"\n\tweights={repr(self.weights)},"
            f"\n\tlocs={repr(self.locs)},"
            f"\n\tscales={repr(self.scales)}"
            f"\n)"
        )

    def sample(self, seed, n):
        rng = np.random.default_rng(seed)
        normal_samples = rng.normal(size=(n,))
        mixture_assignments = rng.choice(self.k, size=n, p=self.weights)
        return normal_samples * self.scales.take(
            mixture_assignments
        ) + self.locs.take(mixture_assignments)
