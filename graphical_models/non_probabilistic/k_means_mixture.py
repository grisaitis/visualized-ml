import jax


class KMeansMixture:
    def __init__(self, means):
        self.k = means.shape[0]
        assert means.shape == (self.k,)
        self.means = means

    def assign(self, x):
        n = x.shape[0]
        distances = jax.numpy.abs(x[:, None] - self.means[None, :])
        assert distances.shape == (n, self.k), distances.shape
        assignment_indices = jax.numpy.argmin(distances, axis=1)
        assert assignment_indices.shape == x.shape, assignment_indices.shape
        assignments = jax.numpy.identity(self.k)[assignment_indices]
        assert assignments.shape == (n, self.k), assignments.shape
        return assignments

    @classmethod
    def learn_with_lloyds_algorithm(cls, key, k_means, x):
        def update(means):
            kmm = cls(means)
            assignments = kmm.assign(x)
            sum_x_per_k = x @ assignments
            assert sum_x_per_k.shape == (k_means,)
            count_per_k = jax.numpy.sum(assignments, axis=0)
            assert count_per_k.shape == (k_means,)
            means = (sum_x_per_k + 1e-8) / (count_per_k + 1e-8)
            assert means.shape == (assignments.shape[1],)
            return means

        means_t_minus_1 = jax.random.normal(key, shape=(k_means,))
        t = 0
        while True:
            t += 1
            print("iteration", t)
            means_t = update(means_t_minus_1)
            print("means", means_t)
            change_magnitude = jax.numpy.linalg.norm(means_t - means_t_minus_1)
            print("change_magnitude", change_magnitude)
            if change_magnitude < 0.0001:
                break
            means_t_minus_1 = means_t

        return cls(means_t)
