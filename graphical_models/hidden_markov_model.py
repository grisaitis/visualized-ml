import numpy as np


class HiddenMarkovModel:
    def __init__(
        self, initial_state_prob, transition_matrix, state_means, state_variances
    ):
        self.k_states = len(initial_state_prob)
        self.initial_state_prob = initial_state_prob
        assert transition_matrix.shape == (self.k_states, self.k_states)
        self.transition_matrix = transition_matrix
        # assume gaussian distribution for observations
        assert state_means.shape == (self.k_states,)
        self.state_means = state_means
        assert state_variances.shape == (self.k_states,)
        assert np.all(state_variances > 0)
        self.state_variances = state_variances

    def generate(self, n_samples, random_state=None):
        rs = random_state or np.random.RandomState()
        states = range(self.k_states)
        y = np.empty(shape=(n_samples,), dtype=np.int)
        x = np.empty(shape=(n_samples,), dtype=np.float)
        y[0] = rs.choice(states, p=self.initial_state_prob)
        for t in range(n_samples - 1):
            y[t + 1] = rs.choice(states, p=self.transition_matrix[y[t]])
        means = np.array([self.state_means[y_t] for y_t in y])
        variances = np.array([self.state_variances[y_t] for y_t in y])
        stdevs = variances ** 0.5
        x = rs.normal(means, stdevs)
        return x, y

    @classmethod
    def learn_from_data(cls, data, n_states):
        return
