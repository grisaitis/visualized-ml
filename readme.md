# graphical models

## how to run examples

i have a few demos under examples/

to run this code you need
- python 3.8 (maybe a version or two less)
- `pip install jax jaxlib numpy`

you can also install dependencies with poetry:
- `poetry install`

and then run examples like
- `python examples/gaussian_mixture/univariate_gmm_with_em.py`

## background

i made this repo while i audit [CMU's class on probabilistic graphical models](https://www.cs.cmu.edu/~epxing/Class/10708-20/) led by Eric Xing. 

Probabilistic graphical modeling in general is study of probabilistic machine learning. It provides a foundation for modeling approaches as diverse as deep learning, bayesian linear regression, clustering, topic modeling (e.g. latent dirichlet allocation), conditional random fields, restricted boltzmann machines (on which deep learning is based), and any other machine learning model that can be expressed in terms of conditional probabilities. 

some things i want to implement and understand better are:
- models like gaussian mixtures, latent variable models (e.g. topic modeling with LDA), hidden markov models
- algorithms for approximate probabilistic inference like variational and monte carlo methods
- algorithms for learning, like expectation maximization
- visualizations of these
