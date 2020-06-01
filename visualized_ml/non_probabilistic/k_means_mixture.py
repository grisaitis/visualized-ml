import jax


class KMeansMixture:
    def __init__(self, means):
        self.k = means.shape[0]
        self.d = means.shape[1] if means.ndim == 2 else None
        self.means = means

    def assign(self, x):
        return _assign(x, self.means)

    @classmethod
    def learn_with_lloyds_algorithm(cls, key, x, k):
        def update(x, means):
            assignments = _assign(x, means)
            return _update_means(x, assignments)

        means_t_minus_1 = _sample_no_replace(key, x, k)
        t = 0
        while True:
            t += 1
            print("iteration", t)
            means_t = update(x, means_t_minus_1)
            print("means", means_t)
            change_magnitude = jax.numpy.linalg.norm(means_t - means_t_minus_1)
            print("change_magnitude", change_magnitude)
            if change_magnitude < 1e-8:
                break
            means_t_minus_1 = means_t
        return cls(means_t)


@jax.jit
def _assign(x, means):
    """assign each x_i to one of the means_k in means"""
    n, k = x.shape[0], means.shape[0]
    distances = jax.numpy.abs(x[:, None] - means[None, :])
    assert distances.shape == (n, k), distances.shape
    assignment_indices = jax.numpy.argmin(distances, axis=1)
    assert assignment_indices.shape == x.shape, assignment_indices.shape
    assignments = jax.numpy.identity(k)[assignment_indices]
    assert assignments.shape == (n, k), assignments.shape
    return assignments


@jax.jit
def _update_means(x, assignments):
    """compute new means given each x_i's value and cluster assignment"""
    n, k = assignments.shape
    sum_x_per_k = x @ assignments
    assert sum_x_per_k.shape == (k,)
    count_per_k = jax.numpy.sum(assignments, axis=0)
    assert count_per_k.shape == (k,)
    means = (sum_x_per_k + 1e-8) / (count_per_k + 1e-8)
    assert means.shape == (k,)
    means = jax.numpy.sort(means)
    return means


def _sample_no_replace(key, x, size):
    """take `size` samples from x without replacement"""
    n = x.shape[0]
    return x[jax.random.shuffle(key, jax.numpy.arange(n))[:size]]


_sample_no_replace = jax.jit(_sample_no_replace, static_argnums=(2,))
