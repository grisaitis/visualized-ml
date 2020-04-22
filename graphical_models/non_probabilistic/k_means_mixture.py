import jax


class KMeansMixture:
    def __init__(self, means):
        self.means = means

    @classmethod
    def learn_from_data(cls, key, k_means, x):
        def e_step(x, means):
            distances = jax.numpy.abs(x[:, None] - means[None, :])
            assert distances.shape == (
                x.shape[0],
                means.shape[0],
            ), distances.shape
            assignment_indices = jax.numpy.argmin(distances, axis=1)
            assert (
                assignment_indices.shape == x.shape
            ), assignment_indices.shape
            assignments = jax.numpy.eye(means.shape[0])[assignment_indices]
            assert assignments.shape == (x.shape[0], k_means)
            return assignments

        def m_step(x, assignments):
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
            assignments_t = e_step(x, means_t_minus_1)
            # print("assignments", assignments_t)
            means_t = m_step(x, assignments_t)
            print("means", means_t)
            assigned_means = assignments_t @ means_t
            assert assigned_means.shape == x.shape
            # print("assignmed_means", assigned_means)
            # print("x", x)
            change_magnitude = jax.numpy.linalg.norm(means_t - means_t_minus_1)
            print("change_magnitude", change_magnitude)
            if change_magnitude < 0.0001:
                break
            means_t_minus_1 = means_t

        return cls(means_t)
