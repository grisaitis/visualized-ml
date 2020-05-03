# graphical models

[![Nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](http://nbviewer.jupyter.org/github/grisaitis/graphical-models/tree/master/) [![Maintainability](https://api.codeclimate.com/v1/badges/014ea0978a4bcbf09c21/maintainability)](https://codeclimate.com/github/grisaitis/graphical-models/maintainability) [![Binder](https://mybinder.org/badge_logo.svg)](http://beta.mybinder.org/v2/gh/grisaitis/graphical-models/master) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grisaitis/graphical-models/blob/master/)

## installation

- python 3.8 (maybe a version or two less)
- dependencies
  - with `pip`: `pip install jax jaxlib numpy`
  - with `poetry`: `poetry install`

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
