import numpy as np
import scipy.stats as stats


class UnivariateGaussianMixture:
    def __init__(self, weights, locs, scales):
        self.k = len(weights)
        assert weights.shape == (self.k,)
        assert locs.shape == (self.k,)
        assert scales.shape == (self.k,)
        assert np.isclose(np.sum(weights), 1.0), (weights, np.sum(weights))
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

    def to_dict(self):
        return dict(
            components=[
                dict(weight=w, loc=l, scale=s)
                for w, l, s in zip(self.weights, self.locs, self.scales)
            ]
        )

    def sample(self, seed, n):
        rng = np.random.default_rng(seed)
        normal_samples = rng.normal(size=(n,))
        mixture_assignments = rng.choice(self.k, size=n, p=self.weights)
        return normal_samples * self.scales.take(
            mixture_assignments
        ) + self.locs.take(mixture_assignments)

    def log_likelihood(self, x):
        prob_xi = self.pdf(x)
        return np.sum(np.log(prob_xi))

    def pdf(self, x):
        n = len(x)
        assert x.shape == (n,)
        pdf_per_component = stats.norm.pdf(
            x[:, None], loc=self.locs, scale=self.scales
        )
        assert pdf_per_component.shape == (n, self.k)
        return pdf_per_component @ self.weights  # (n, k) @ (k,)
