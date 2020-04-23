from graphical_models.hidden_markov_model import HiddenMarkovModel

import numpy as np


transition = np.array([[0.8, 0.2], [0.4, 0.6]])
initial_p = np.array([0.9, 0.1])
means = np.array([-10, 10])
variances = np.array([1, 2])

hmm = HiddenMarkovModel(initial_p, transition, means, variances)

rs = np.random.RandomState(0)

x, y = hmm.generate(n_samples=10, random_state=rs)

print(y)
print(x)

y_inferred_true_model = hmm.infer_most_probable_y(x=x)

print(y_inferred_true_model)

hmm_learned = HiddenMarkovModel.learned_from_data(data=x, n_states=2)

print(hmm_learned.means)
print(hmm_learned.variances)
print(hmm_learned.initial_p)
print(hmm_learned.transition)

y_inferred = hmm_learned.infer_most_probable_y(x=x)

print(y_inferred)
