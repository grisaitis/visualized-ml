# visualized-ml

[![Nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](http://nbviewer.jupyter.org/github/grisaitis/visualized-ml/tree/master/) [![Maintainability](https://api.codeclimate.com/v1/badges/014ea0978a4bcbf09c21/maintainability)](https://codeclimate.com/github/grisaitis/visualized-ml/maintainability) [![Binder](https://mybinder.org/badge_logo.svg)](http://beta.mybinder.org/v2/gh/grisaitis/visualized-ml/master) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grisaitis/visualized-ml/blob/master/)

## background

in this project i'm doing two things:
- implementing some machine learning models and training algorithms from scratch
- visualizing these these algorithms in realtime with logging and plotting libraries

i first made this repo when i started auditing a class on probabilistic graphical models: [CMU 10708-20](https://www.cs.cmu.edu/~epxing/Class/10708-20/) by [Prof. Eric Xing](https://www.cs.cmu.edu/~epxing/).

### what are "probabilistic graphical models"?

Probabilistic graphical models are (don't quote me on this) models that use conditional probability distributions to model relationships between variables in data. Models are represented as graphs, with variables as nodes and relationships (e.g. conditional dependence) as edges or arcs (directed edges). It's a framework for defining and analyzing probabilistic models and algorithms on them. It provides a foundation for modeling approaches as diverse as kernel density estimation, linear regression, deep learning, topic modeling (e.g. latent dirichlet allocation), conditional random fields, restricted boltzmann machines (on which deep learning is based), and any other machine learning model that can be expressed in terms of probability distributions.

## my goals

- implement and understand the following:
  - probabilistic models (e.g. gaussian mixtures, hidden markov)
  - learning / optimization algorithms (e.g. expectation maximization, various gradient or coordinate descent methods)
  - inference algorithms (monte carlo methods, variational methods, maybe exact methods like Viterbi for hidden markov models)
- visualize these models and algorithms, to better understand (and debug) them
